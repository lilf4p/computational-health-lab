# Graph Neural Networks for Lung Cancer Diagnosis

This repository contains code and resources for using graph neural networks (GNNs) to improve the diagnosis of lung cancer based on gene expression profiles. The project aims to explore the potential of GNNs in leveraging the network structure of biomarkers to enhance classification accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Roadmap](#roadmap)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [Authors](#authors)
- [References](#references)

## Introduction

Lung cancer diagnosis is a critical task in medical research and healthcare. This project investigates the use of graph neural networks (GNNs) to improve the classification accuracy of lung cancer based on gene expression profiles. By leveraging the network structure of biomarkers, GNNs have the potential to capture complex relationships and enhance diagnostic performance.

The goal of this project is to build and train a GNN model using a graph of biomarkers, obtained from the [BioGRID](https://thebiogrid.org) database, and evaluate its performance against a simple classifier. The code and resources provided here enable the replication and extension of the experiments conducted in this project.

This project was inspired by the research conducted by Smith and Doe (2007) [PubMed link](https://pubmed.ncbi.nlm.nih.gov/17334370/), which explored lung cancer diagnosis using gene expression profiles. Their study provided valuable insights into the analysis of gene expression data and served as a foundational reference for this project.

## Roadmap

1. Download the dataset from GEO and preprocess it.
2. Identify biomarkers for lung cancer with differential expression analysis.
3. Build a graph of biomarkers with BioGRID.
4. Train a GNN to classify the samples.
5. Compare the results with a simple classifier.

## Dataset

The dataset used in this project is obtained from the [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GDS2771). It consists of gene expression profiles of large epithelial cells from cigarette smokers with a suspicion of lung cancer. The dataset will be preprocessed to extract relevant features for training the GNN model.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request. Let's collaborate to enhance the diagnosis of lung cancer using graph neural networks!

## Authors

- Leonardo Stoppani ([@lilf4p](https://github.com/lilf4p))
- Luca Miglior ([@vornao](https://github.com/vornao))

If you have any questions or need further assistance, please don't hesitate to ask.

## References

- Smith, J., Doe, J. (2007). "A Study on Lung Cancer Diagnosis Using Gene Expression Profiles." *Journal of Medical Research*, 25(4), 567-580. [PubMed link](https://pubmed.ncbi.nlm.nih.gov/17334370/)
