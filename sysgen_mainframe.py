import pandas as pd

from data_collection import build_gene_location_dict, gene_expression_pipeline
from qtl_analysis import snp_regression

CHOSEN_TISSUES = {
    'liver': 'GSE17522',
    'kidney': 'GSE8356',
}
GENOTYPE_PATH = './expression_data/genotypes.xls'
PHENOTYPE_PATH = './expression_data/phenotypes.xls'


def genotype_preprocessing(genotype_data, studied_strains):
    """
    Limits a genotype to a subset of BXD strains, and a representative locus for
    each stretch of SNPs which have the exact same profile (same b,h,d allele for
    each BXD strain).
    """
    genotypes = genotype_data[[col for col in genotype_data.columns if
                               not col.startswith('BXD') or col in studied_strains]]
    representative_loci = []
    loci_range = []
    current_chromosome = 0
    current_profile = None

    for row_name, row_data in genotypes.iterrows():
        row_chromosome, row_profile = row_data["Chr_Build37"], row_data.filter(regex="BXD[0-9]+")

        if row_chromosome == current_chromosome and row_profile.equals(current_profile):
            loci_range.append(row_name)
        else:
            if loci_range:
                representative_loci.append(loci_range[len(loci_range) // 2])
            current_chromosome, current_profile = row_chromosome, row_profile
            loci_range = [row_name]
    return genotypes.loc[representative_loci]


def phenotype_preprocessing(phenotype_data, studied_strains, drop_na=True):
    phenotype_cols = phenotype_data[[col for col in phenotype_data.columns if
                                     not col.startswith('BXD') or col in studied_strains]]
    if drop_na:
        phenotype_cols = phenotype_cols.dropna(axis=0, how='any')
    return phenotype_cols


if __name__ == "__main__":
    gene_locations = build_gene_location_dict()
    gene_expression = {tissue: gene_expression_pipeline(geo_id, tissue, gene_locations) for
                       tissue, geo_id in CHOSEN_TISSUES.items()}

    expression_strains = gene_expression['liver'].columns[
        gene_expression['liver'].columns.str.startswith('BXD')]
    genotype_raw = pd.read_excel(GENOTYPE_PATH, header=0, index_col=0, skiprows=[0])
    genotype_data = genotype_preprocessing(genotype_raw, expression_strains)
    phenotype_raw = pd.read_excel(PHENOTYPE_PATH, header=0, index_col=0)
    phenotype_data = phenotype_preprocessing(phenotype_raw, expression_strains, drop_na=False)

    for phenotype_name, phenotype_row in phenotype_data.iterrows():
        threshold = 0.05 / len(genotype_data.index)
        p_values = []
        for snp_name, snp_row in genotype_data.iterrows():
            try:
                p_values.append(snp_regression(snp_row, phenotype_row))
            except AssertionError:
                print("--| Can't perform regression for: ", snp_name, phenotype_name)
            except ZeroDivisionError:
                print("--| some other error for: ", snp_name, phenotype_name)
        print(phenotype_name, min(p_values), any([pv <= threshold for pv in p_values]))
