A collection of scripts to extract data from the MovieLens dataset and reproduce the experiments described in the paper.

1. PREREQUISITES  

   * Python ≥ 3.10  
   * Ensure you have the following files in one folder:    
     • sol.py  
     • data_extraction.py  
     • MovieLens files: u.data, u.user  
     • Precomputed “top_xx” files: top_10, top_20, top_50, top_75, top_100  
     • requirements.txt  

2. INSTALLATION   

   To install the dependencies use the following command

   ```bash
   pip install –r requirements.txt
   ```

3. DATA EXTRACTION  
   Use data_extraction.py to preprocess the MovieLens ratings,movie and user files:  
   ```bash 
   python3 data_extraction.py < u.data > input
   ```

   Configuration within data_extraction.py:
   • top_n: top n movies we want to consider
   • Utility threshold

5. RUNNING EXPERIMENTS  
   Your main driver is sol.py. To reproduce the paper’s experiments:  
   ```bash
   python3 sol.py < top_xx
   ```
   Replace xx with one of: 10, 20, 50, 75, or 100  
   Ensure the corresponding top_xx file is in the same folder as sol.py  

7. FILE OVERVIEW  
   sol.py            Implements all algorithms from the paper.    
   data_extraction.py  Parses u.data and u.user and produces the input file.    
   top_xx            Precomputed input subsets (top-10, -20, -50, -75, -100) for sol.py.    

8. GENERATING PLOTS
   If you want to generate plots, you have to run `rounding_while_plotting` instead of the default rounding function. To do this, you need to modify `sol.py`. In the main execution block (at the end of the file), change the function call from `rounding` to `rounding_while_plotting`.  

   Specifically, change this line:  
   `OPT=rounding(network,lp(network,'s','t',l,uu),num_items,uu)`  

   To this:  
   `OPT=rounding_while_plotting(network,lp(network,'s','t',l,uu),num_items,uu)`  

   Plots in paper are generated through
   ```python
   def f(k):
      x=k
      if(k==0):
         return 1
      else:
         return -math.log(x)

   def g(k):
      x=k
      if(k==0):
         return 1
      else:
         return -math.log(x)
   ```

   Make sure to change this before running
