# PyTorch Factorized_Personalized_Markov_Chains
Implemented "Factorizing Personalized Markov Chains for Next-Basket Recommendation" for next item prediction for Movie Lens 1 million

Dataset for Movie Lens 1-million may be downloaded from https://grouplens.org/datasets/movielens/1m/

`python FPMC_Optimization.py --num_epochs=250 --alpha=0.001 --UI_dim=128 --IL_dim=128 read_filename='ml-1m\\ratings.dat'`  

**Argument Descriptions**:  
  --num_epochs: type=int, help='Number of Training Epochs', default=250  
  --alpha: type=float, help='Learning Rate', default=1e-3  
  --UI_dim: type=int,help="Size of User-Item interaction embedding dimension for matrix factorization",default=64  
  --IL_dim: type=int,help="Size of Next Item - Last Item interaction embedding dimension for matrix factorization",default=64  
  --read_filename: type=str,help='The filename to read all the MovieLens-1 million data from to the Dataframe',default="ml-1m\\ratings.dat"  
  --batch_size: type=int,help='The batch size for stochastic gradient descent',default=512  
  --reg': type=float,help='The regularization strength on l2 norm',default = 0.00005  
  --hitsat': type=int,help='The number of items to measure the hit@k metric (i.e. hit@10 to see if the correct item is within the top 10 scores)',default=10  
  --evaluate_every: type=int,help='How frequent (epochs) should you print out the validation and test recall@k metric',default=15  
  --print_every: type=int,help='How frequent (stochastic gradient updates) should you print out the running training loss',default=250  

## Match Hit@10 test result for ML-1m of about ~58% without hyperparameter search compared to result reported in Bert4Rec (~59%)  
