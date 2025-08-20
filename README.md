# BMSSP: Batch-Minimum Shortest Path Algorithm in Python  

🚀 A Python implementation of the **BMSSP (Batch-Minimum Single Source Shortest Path)** algorithm introduced in *[Batch-Minimum Single Source Shortest Paths (arXiv:2504.17033, 2025)](https://arxiv.org/abs/2504.17033)*.  

---

## ✨ Features  
- ✅ **BMSSP implementation** with recursive relaxation and multi-phase scheduling  
- ✅ **Robust driver system** with automatic fallback to Dijkstra for guaranteed correctness  
- ✅ **Random graph generator** supporting weighted directed graphs (up to 10k nodes, 100k edges)  
- ✅ **Graph visualization suite** using NetworkX + Matplotlib with dynamic layouts, colored paths, and edge weights  
- ✅ **Benchmarking harness** to compare BMSSP vs. classical algorithms (Dijkstra, Bellman-Ford)  

---

## 📖 Background  
The **BMSSP algorithm** improves shortest path computation by batching relaxations and progressively converging on correct distances. Unlike traditional algorithms:  

- **Dijkstra**: Processes vertices one at a time with a priority queue.  
- **Bellman-Ford**: Relaxes all edges repeatedly, O(VE) complexity.  
- **BMSSP**: Uses *batches of minimum relaxations* to accelerate convergence while avoiding redundant updates.  

This repo provides a faithful implementation of the algorithm and evaluates its performance on synthetic graphs.  

---

## 🔧 Installation  

```bash
git clone https://github.com/yourusername/bmssp.git
cd bmssp
pip install -r requirements.txt
```
## Usage
python main.py --nodes 50 --edges 200 --source 0
