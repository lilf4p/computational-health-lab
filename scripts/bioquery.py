# this script will query biogrid database for protein-protein interactions
# and output the results in a tab-delimited format

import requests
import json
import pandas as pd

base_url = "https://webservice.thebiogrid.org/"
includeInteractors = "true"
gene = ""
key = ""

# define the base url
url = f"{base_url}interactions?searchNames=true&geneList={gene}&includeInteractors={includeInteractors}&includeInteractorInteractions=false&taxId=9606&format=json&accesskey={key}"


# each result is a json object, containing other objects.
# each object has a key, the BioGrid ID. The value is a dict with the following keys:
# "119730": {                                                // this is the BioGrid ID
#       "BIOGRID_INTERACTION_ID": 119730,
#       "ENTREZ_GENE_A": "9391",
#       "ENTREZ_GENE_B": "79741",
#       "BIOGRID_ID_A": 114791,
#       "BIOGRID_ID_B": 122854,
#       "SYSTEMATIC_NAME_A": "-",
#       "SYSTEMATIC_NAME_B": "RP11-195O1.4",                // again may be useful to find the gene name in our dataset
#       "OFFICIAL_SYMBOL_A": "CIAO1",
#       "OFFICIAL_SYMBOL_B": "CCDC7",                       // this is the interacting gene name?
#       "SYNONYMS_A": "CIA1|WDR39",                         // synonims of our gene name
#       "SYNONYMS_B": "BioT2-A|BioT2-B|BioT2-C|C10orf68",   // these may be useful to find the interacting gene name in our dataset
#       "EXPERIMENTAL_SYSTEM": "Two-hybrid",                // this is the type of experiment
#       "EXPERIMENTAL_SYSTEM_TYPE": "physical",             // this is the type of evidence (quello che diceva priami)
#       "PUBMED_AUTHOR": "Rual JF (2005)",
#       "PUBMED_ID": 16189514,
#       "ORGANISM_A": 9606,                                 // homo sapiens
#       "ORGANISM_B": 9606,                                 // homo sapiens
#       "THROUGHPUT": "High Throughput",
#       "QUANTITATION": "-",
#       "MODIFICATION": "-",
#       "ONTOLOGY_TERMS": {},
#       "QUALIFICATIONS": "-",
#       "TAGS": "-",
#       "SOURCEDB": "BIOGRID"
#   }, ...                                                  // other objects following
# we may want to keep only commented fields.
# what is the difference among interactions and interactors?


def query_biogrid(gene_name, api_key):
    # define the base url
    url = f"{base_url}interactions?searchNames=true&geneList={gene_name}&includeInteractors={includeInteractors}&includeInteractorInteractions=false&taxId=9606&format=json&accesskey={api_key}"
    # check for errors
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print("> ERROR: Failed to get {gene_name}")
        return 0


def parse_interactions(interactions: dict) -> set:
    """this method will take a response and return a list of interactions and their synonims"""
    i = set()
    for key in interactions:
        i.add(interactions[key]["OFFICIAL_SYMBOL_B"])
        # get synonims b and separate them
        synonims = interactions[key]["SYNONYMS_B"].split("|")
        for synonim in synonims:
            i.add(synonim)

    return i


def find_interactions(gene_list, api_key) -> pd.DataFrame:
    # for each gene in deExpression.csv query biogrid.
    # if the response contains a gene present in deExpression, then add it do adj matrix
    # adj matrix is a pandas dataframe with genes as rows and columns
    # if gene is present in deExpression, then add 1 to the corresponding cell

    adj_matrix = pd.DataFrame(0, index=gene_list, columns=gene_list)

    for gene in gene_list:
        print("> Querying Biogrid for gene: ", gene)
        interactions = query_biogrid(gene, api_key)
        if interactions:
            intset = parse_interactions(interactions)
            for i in intset:
                if i in gene_list:
                    adj_matrix.at[gene, i] = 1
    return adj_matrix
