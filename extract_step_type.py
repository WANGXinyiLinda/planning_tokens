from tqdm import tqdm
import json, pickle
import collections, time
import argparse
from typing import Optional
import numpy as np
import transformers
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sentence_transformers
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from transformers import T5EncoderModel

from load_data.preprocess import *
from load_data.supervised_dataset import SupervisedDataset, DataCollatorForSupervisedDataset
from model.vae import VAE

def phi1_forward(
    self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, 
    past_key_values: Optional[torch.FloatTensor] = None, **kwargs
) -> CausalLMOutputWithPast:

    hidden_layer = self.layers[0](input_ids)
    for module in self.layers[1:-1]:
        hidden_layer = module(hidden_layer, past_cache=past_key_values if past_key_values is not None else None)
    lm_logits = self.layers[-1](hidden_layer)

    loss = None
    if labels is not None:
        loss = self.loss(lm_logits, labels)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=past_key_values,
        hidden_states=[hidden_layer],
    )


def extract_step_type(dataset_name:str, model_name_or_path:str, batch_size:int,
                      model_max_length = 1024, train_epoch=10,
                      selection_method='vae',
                      output_dir='extract_steps', cache_dir=None,
                      min_frequency=10, max_frequency=10000, num_types=50,):
    
    if model_name_or_path != 'all-mpnet-base-v2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
        tokenizer.model_max_length = model_max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = 0

    if dataset_name == "gsm8k":
        data_class = GSMData
    elif dataset_name == "math":
        data_class = MathData
    elif dataset_name == "aqua":
        data_class = AquaData
    elif dataset_name == "svamp":
        data_class = SVAMPData
    else:
        raise NotImplementedError
    
    dataset = data_class("train", {})

    if selection_method == 'tf-idf':
        
        solution_steps = []
        for d in dataset:
            solution_steps += d['y'].split('\n')[:-1]

        solution_steps = [step.strip() for step in solution_steps if len(step.strip())>0 and "The answer is:" not in step]

        save_file = f"{output_dir}/{dataset_name}_{selection_method}.pkl"

        if os.path.isfile(save_file):
            with open(save_file, 'rb') as f:
                vectorizer = pickle.load(f)
            X = vectorizer.transform(solution_steps)
        else:
            vectorizer = TfidfVectorizer(max_df=max_frequency, min_df=min_frequency, tokenizer=word_tokenize)
            X = vectorizer.fit_transform(solution_steps)
            print(X)
            with open(save_file, 'wb') as f:
                pickle.dump(vectorizer, f)

        vocab = vectorizer.get_feature_names_out()
        print(vocab)
        print("num vocab: ", len(vocab))
        ids = np.argmax(X, axis=1)
        step_types = []
        for i in ids:
            step_types.append(vocab[i][0][0])
        assert len(step_types) == len(solution_steps)
        all_step_types = set(step_types)
        step_type_count = {}
        for step_type in all_step_types:
            step_type_count[step_type] = step_types.count(step_type)
        print({key: val for key, val in sorted(step_type_count.items(), key = lambda ele: ele[1], reverse = True)})
        print(all_step_types)
        print('num all step types: ', len(all_step_types))

    elif any(m in selection_method for m in ['vae', 'k-means']):

        solution_steps = []
        questions = []
        for d in dataset:
            questions.append(d['x'].strip().split('\n'))
            steps = d['y'].strip().split('\n')
            if len(steps) > 1:
                steps = [step.strip() for step in steps if len(step.strip())>0]
                solution_steps.append(steps[:-1])
        
        out_dir = f"{output_dir}/{model_name_or_path}/{dataset_name}"
        embedding_file = f"{out_dir}/{dataset_name}_embedding.pkl"

        if os.path.isfile(embedding_file):
            with open(embedding_file, 'rb') as f:
                step_embeddings = pickle.load(f)
                print(step_embeddings)

        else:
            if model_name_or_path == 'all-mpnet-base-v2':
                embedding_model = sentence_transformers.SentenceTransformer(model_name_or_path)

            elif 't5' in model_name_or_path:
                embedding_model = T5EncoderModel.from_pretrained(model_name_or_path).to('cuda')

            else:
                embedding_model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, cache_dir=cache_dir).to('cuda')
                
                if 'phi-1' in model_name_or_path:
                    # monkey patching to add output hidden states
                    from functools import partial

                    embedding_model.forward = partial(phi1_forward, self=embedding_model)

            embedding_model.eval()

            os.makedirs(out_dir, exist_ok=True)
            step_embeddings = []
            with torch.no_grad():
                if 't5' in model_name_or_path:
                    for steps in tqdm(solution_steps):
                        inputs = tokenizer(steps, return_tensors="pt", padding="longest",
                            max_length=model_max_length, truncation=True).to('cuda')
                        outputs = embedding_model(**inputs)
                        last_hidden_states = outputs.last_hidden_state
                        mean_hidden = torch.sum(last_hidden_states 
                            * inputs["attention_mask"].unsqueeze(-1), 1)/torch.sum(inputs["attention_mask"], -1).unsqueeze(-1)
                        step_embeddings.append(mean_hidden.cpu().numpy())

                if 'llama' in model_name_or_path or 'gpt2' in model_name_or_path:
                    dataset = SupervisedDataset(dataset, tokenizer)
                    data_collator = DataCollatorForSupervisedDataset(tokenizer, [], False)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                            collate_fn=data_collator)

                    for i, batch in tqdm(enumerate(dataloader)):
                        for k in batch:
                            batch[k] = batch[k].to('cuda')
                        outputs = embedding_model(**batch, output_hidden_states=True, 
                                        return_dict=True)
                        last_hidden_states = outputs.hidden_states[-1]
                        if 'llama' in model_name_or_path:
                            split_id = 13
                        elif 'gpt2' in model_name_or_path:
                            split_id = 198
                        step_mask = torch.cumsum(batch['input_ids']==split_id, dim=-1)
                        text_batch = solution_steps[i*batch_size:(i+1)*batch_size]
                        q_batch = questions[i*batch_size:(i+1)*batch_size]
                        print(q_batch)
                        print(text_batch)
                        print(step_mask)
                        print([len(sents) for sents in text_batch])
                        for hidden, mask, q in zip(last_hidden_states, step_mask, q_batch):
                            example_rep = []
                            start = min(len(q), torch.max(mask)-1)
                            print(start)
                            for j in range(start, torch.max(mask)-1):
                                step_j_mask = (mask == j).int().float()
                                step_j_rep = (hidden * step_j_mask.unsqueeze(-1)).sum(0)
                                step_len = step_j_mask.sum()
                                if step_len > 0:
                                    example_rep.append((step_j_rep/step_len).cpu().numpy())
                                else:
                                    print("current step is empty")
                            if len(example_rep) > 0:
                                example_rep = np.stack(example_rep, axis=0)
                                step_embeddings.append(example_rep)

                else:
                    for steps in tqdm(solution_steps):
                        step_embeddings.append(embedding_model.encode(steps))

            with open(embedding_file, 'wb') as f:
                pickle.dump(step_embeddings, f)
        
        out_dir = f"{out_dir}/{selection_method}-{num_types}"
        solution_steps = [step for steps in solution_steps for step in steps]

        if 'k-means' in selection_method:
            cluster_model_file = f"{out_dir}/{dataset_name}_{selection_method}_{num_types}.pkl"
            if os.path.isfile(cluster_model_file):
                with open(cluster_model_file, 'rb') as f:
                    cluster_model = pickle.load(f)
            else:
                os.makedirs(f"{out_dir}", exist_ok=True)
                if 'balanced' in selection_method:
                    # To install, see https://github.com/kernelmachine/balanced-kmeans/tree/main
                    from kmeans_pytorch import KMeans as BalancedKMeans

                    step_embeddings = torch.from_numpy(np.float32(np.concatenate(step_embeddings, axis=0))).cuda()
                    cluster_model = BalancedKMeans(
                        n_clusters=num_types, 
                        device='cuda', 
                        balanced=True, 
                    )
                    indices = cluster_model.fit(step_embeddings, iter_limit=300, tol=0.).numpy()

                    # wrap it inside the regular k-means module to piggyback on existing code
                    dummy_input = step_embeddings[:10].cpu().numpy()
                    sklearn_cluster_model = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(dummy_input)
                    sklearn_cluster_model.cluster_centers_ = cluster_model.cluster_centers.cpu().numpy()

                    assert np.all(
                        sklearn_cluster_model.predict(step_embeddings.cpu().numpy()) ==
                        cluster_model.predict(step_embeddings).cpu().numpy()
                    )
                    cluster_model = sklearn_cluster_model

                else:
                    step_embeddings = np.float32(np.concatenate(step_embeddings, axis=0))
                    cluster_model = KMeans(n_clusters=num_types, n_init=10, random_state=0).fit(step_embeddings)
                    indices = cluster_model.labels_
            
                with open(cluster_model_file, 'wb') as f:
                    pickle.dump(cluster_model, f)

            assert len(indices) == len(solution_steps)

            solution_steps = np.array(solution_steps)
            for i in range(num_types):
                print(f"cluster {i}: ", np.sum(indices==i))
                with open(f"{out_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                    f.write('\n'.join(list(solution_steps[indices==i])))

        elif 'vae' in selection_method:
            solution_steps = np.array(solution_steps)
            cluster_model_dir = f"{out_dir}/epoch{train_epoch-1}"
            cluster_model_file = f"{out_dir}/epoch{train_epoch-1}/{dataset_name}_{selection_method}_{num_types}.pkl"
            if os.path.isfile(cluster_model_file):
                print("model exists, loading...")
                with open(cluster_model_file, 'rb') as f:
                    cluster_model = pickle.load(f)

                print("inspecting clusters...")
                indices = []
                cluster_model.eval()
                with torch.no_grad():
                    for input_batch in tqdm(step_embeddings):
                        input_batch = torch.from_numpy(input_batch).float().to('cuda')
                        idx = cluster_model.predict(input_batch)
                        indices.append(idx.cpu().numpy())
                indices = np.concatenate(indices, axis=0)

                assert len(solution_steps) == len(indices)

                for i in range(num_types):
                    print(f"cluster {i}: ")
                    print(np.sum(indices == i))
                    with open(f"{cluster_model_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                        f.write('\n'.join(list(solution_steps[indices == i])))
            else:
                os.makedirs(f"{out_dir}", exist_ok=True)
                input_dim = step_embeddings[0].shape[1]
                print("input embedding dim: ", input_dim) 
                neg_cost = 0
                if 'next-step' in selection_method:
                    loss_type = 'next-step'
                else:
                    loss_type = 'contrastive'
                    if 'contrastive' in selection_method:
                        neg_cost = 0.1

                cluster_model = VAE(input_size=input_dim, num_embeddings=num_types,
                                    neg_cost=neg_cost, loss_type=loss_type).to('cuda')
                checkpoint_vals = collections.defaultdict(lambda: [])

                for epoch in range(train_epoch):
                    epoch_start_time = time.time()
                    cluster_model_dir = f"{out_dir}/epoch{epoch}"
                    cluster_model_file = f"{out_dir}/epoch{epoch}/{dataset_name}_{selection_method}_{num_types}.pkl"
                    os.makedirs(cluster_model_dir, exist_ok=True)
                    print(f"training epoch {epoch+1}...")
                    np.random.shuffle(step_embeddings)
                    for input_batch in tqdm(step_embeddings):
                        input_batch = torch.from_numpy(input_batch).float().to('cuda')
                        losses = cluster_model.update(input_batch)
                        for key, val in losses.items():
                            checkpoint_vals[key].append(val)
                    print(f"epoch {epoch+1} finished, time: {time.time()-epoch_start_time}")
                    checkpoint_vals['epoch_time'].append(time.time() - epoch_start_time)

                    print("saving epoch checkpoint...")
                    results = {'epoch': epoch+1}
                    for key, val in checkpoint_vals.items():
                        results[key] = np.mean(val)
                    print(results)
                    results['args'] = vars(args)

                    epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                    with open(epochs_path, 'w') as f:
                        f.write(json.dumps(results, sort_keys=True) + "\n")

                    checkpoint_vals = collections.defaultdict(lambda: [])
                    with open(cluster_model_file, 'wb') as f:
                        pickle.dump(cluster_model, f)

                    print("inspecting clusters...")
                    indices = []
                    cluster_model.eval()
                    with torch.no_grad():
                        for input_batch in tqdm(step_embeddings):
                            input_batch = torch.from_numpy(input_batch).float().to('cuda')
                            idx = cluster_model.predict(input_batch)
                            indices.append(idx.cpu().numpy())
                    indices = np.concatenate(indices, axis=0)

                    assert len(solution_steps) == len(indices)

                    for i in range(num_types):
                        print(f"cluster {i}: ")
                        print(np.sum(indices == i))
                        with open(f"{cluster_model_dir}/{dataset_name}_{num_types}_{i}.txt", 'w') as f:
                            f.write('\n'.join(list(solution_steps[indices == i])))


        tsne_file = f"{output_dir}/{selection_method}/{dataset_name}/{dataset_name}_{selection_method}_tsne.npy"

        if os.path.isfile(tsne_file):
            X = np.load(tsne_file)
        else:
            X = TSNE(n_components=2, learning_rate='auto',
                            init='random', perplexity=3).fit_transform(np.concatenate(step_embeddings))
            np.save(tsne_file, X)

        plt.scatter(X[:, 0], X[:, 1], c=indices, s=2, cmap='viridis')
        plt.title(f"Number of Clusters = {num_types}")
        plt.savefig(f"{output_dir}/{selection_method}/{dataset_name}/{num_types}/{dataset_name}_kmeans.png")
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gsm8k', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf', help='model name or path')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_max_length', type=int, default=2048, help='model max length')
    parser.add_argument('--selection_method', type=str, default='diff_mean', 
                        choices=['tf-idf', 'k-means', 'vae-next-step', 'vae-contrastive',
                                 'vae', 'balanced-k-means'], help='selection method')
    parser.add_argument('--output_dir', type=str, default='load_data/extract_steps', help='output dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='cache dir')
    parser.add_argument('--min_frequency', type=int, default=0.05, help='min frequency')
    parser.add_argument('--max_frequency', type=int, default=0.80, help='max frequency')
    parser.add_argument('--num_types', type=int, default=50, help='number of reasoning types')
    parser.add_argument('--train_epoch', type=int, default=10, help='number of training epochs')

    args = parser.parse_args()

    extract_step_type(args.dataset, args.model_name_or_path, args.batch_size,
                      args.model_max_length, args.train_epoch,
                      args.selection_method, args.output_dir, 
                      args.cache_dir, args.min_frequency, args.max_frequency,
                      args.num_types)