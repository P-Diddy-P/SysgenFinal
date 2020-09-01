import re
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import GEOparse as geo
from Bio import Entrez

Entrez.api_key = 'cc86c67528c5c05e90be06cace971d287b08'
Entrez.email = 'omershapira@mail.tau.ac.il'
Entrez.tool = 'sysgen-final-project'
ENTREZ_BATCH_SIZE = 15


def extract_gene_location(gene_row):
    """
    returns the first non-null location of a gene (rep., NCBI or Ensembl),
    if no location is found, returns a default dictionary
    """
    gene_location = dict()
    if not any(pd.isna(gene_row.iloc[4:7])):
        gene_location['chromosome'] = gene_row.iloc[4]
        gene_location['start'] = gene_row.iloc[5]
        gene_location['end'] = gene_row.iloc[6]
        return gene_location
    if not any(pd.isna(gene_row.iloc[10:13])):
        gene_location['chromosome'] = gene_row.iloc[10]
        gene_location['start'] = gene_row.iloc[11]
        gene_location['end'] = gene_row.iloc[12]
        return gene_location
    if not any(pd.isna(gene_row.iloc[15:18])):
        gene_location['chromosome'] = gene_row.iloc[15]
        gene_location['start'] = gene_row.iloc[16]
        gene_location['end'] = gene_row.iloc[17]
    return gene_location


def build_gene_location_dict():
    """
    Searches for an intermediate file with easily accessed gene locations.
    If no such file exists, builds the file and saves it as an intermediate.
    """
    if 'gene_locations.csv' in os.listdir('./intermediate_files'):
        print("Intermediate gene locations file found. Retrieving...")
        gene_locations = pd.read_csv('./intermediate_files/gene_locations.csv', sep=',')
        return gene_locations.set_index('marker symbol')

    print("No intermediate gene location file. Building...")
    raw_location = pd.read_csv(
        './MGI_Coordinates.Build37.rpt.txt',
        delimiter='\t',
        index_col=0,
        usecols=[i for i in range(20)]
    )

    gene_locations = {'chromosome': [], 'start': [], 'end': []}
    for _, row in raw_location.iterrows():
        loc = extract_gene_location(row)
        gene_locations['chromosome'].append(loc.get('chromosome', np.nan))
        gene_locations['start'].append(loc.get('start', np.nan))
        gene_locations['end'].append(loc.get('end', np.nan))

    compact_location = pd.concat(
        [raw_location['marker symbol'], raw_location['marker name'], raw_location['marker type']],
        axis=1,
        keys=['marker symbol', 'marker name', 'marker type']
    ).assign(**gene_locations).dropna(axis=0, how='any', inplace=False)
    compact_location = compact_location[compact_location['marker type'] == 'Gene']
    compact_location.set_index('marker symbol', inplace=True)
    compact_location.to_csv('./intermediate_files/gene_locations.csv')
    return compact_location


def get_gsm_bxd_name(gsm_record, specifier=''):
    """
    Given a GSM (GEO Sample) record, extract the sample's BXD strain name and return it,
    if no strain is found, return empty string. This function also takes care of substrains,
    such as BXD26a, regardless of the desired course of action with such substrains.

    * For hematopoietic samples we will only consider Erythroid expression samples.
    """
    bxd_pattern = r'BXD[1-9][0-9]{0,2}[a-zA-Z]?'
    batch_pattern = '(?P<type>[A-Z][a-z]+[0-9]?) batch[0-9]'
    gsm_title = gsm_record.metadata['title'][0]
    bxd_match = re.search(bxd_pattern, gsm_title)
    batch_match = re.search(batch_pattern, gsm_title)

    if bxd_match and not batch_match:
        return bxd_match.group()
    if bxd_match and batch_match and batch_match.group('type') == specifier:
        return bxd_match.group()
    return ''


def generate_raw_expression_table(gse):
    """
    Given a GEO GSE, returns an initial datatable, before any averaging or
    dropping of data reads.

    Multiple columns of the same strain will receive a numbered suffix by looking
    up previous gsm name occurances in the `name_table`.
    """
    tables = []
    for gpl in gse.gpls.values():
        base_table = gpl.table[['ID', 'GB_ACC']]
        read_table = dict()
        name_table = dict()
        for gsm in gse.gsms.values():
            gsm_name = get_gsm_bxd_name(gsm, 'Erythroid')

            if gsm_name:
                insert_name = gsm_name
                if gsm_name in name_table:
                    insert_name += ('.' + name_table[gsm_name])

                read_table[insert_name] = gsm.table['VALUE']
                name_table[gsm_name] = name_table.get(gsm_name, 0) + 1

        tables.append(base_table.assign(**read_table))

    assert len(tables) < 2
    return tables[0]


def extract_gene_name(entrez_xml):
    """
    Given a JSON-like Entrez.read result, searches the entry's features for a gene feature.
    If a gene feature is found, return it's qualified name. If no qualifier is found, return
    an empty string
    """
    for feature in entrez_xml['GBSeq_feature-table']:
        if feature['GBFeature_key'] != 'gene':
            continue
        for qualifier in feature['GBFeature_quals']:
            if qualifier['GBQualifier_name'] == 'gene':
                return entrez_xml['GBSeq_locus'], qualifier['GBQualifier_value']
    return '', ''


def accession_get_gene(acc_data, dest_dict, loc_dict):
    """
    Requests gene names for a batch of accession ids, and adds a their
    location data from the already processed location dict.
    """
    acc_data = {v: k for k, v in acc_data.items()}
    response = Entrez.efetch(db='nuccore', retmode='xml',
                             id=','.join(acc_data.keys()))
    for entrez_entity in Entrez.parse(response):
        locus_name, gene_name = extract_gene_name(entrez_entity)
        if gene_name in loc_dict and locus_name in acc_data:
            dest_dict[acc_data[locus_name]] = [gene_name]
            dest_dict[acc_data[locus_name]].extend(loc_dict[gene_name])
            # print("{0}: {1}".format(acc_data[locus_name],
            #                         dest_dict[acc_data[locus_name]]))


def get_table_genes(gene_table, location_dict):
    """
    Given a gene table, batches ids and accession ids, and
    runs a thread pool to annotate all genes in batches
    """
    genes_by_id = dict()
    acc_id = dict()
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _, row in gene_table.iterrows():
            acc_id[row['ID']] = row['GB_ACC']
            if len(acc_id) >= ENTREZ_BATCH_SIZE:
                executor.submit(
                    accession_get_gene,
                    acc_data=dict(acc_id),
                    dest_dict=genes_by_id,
                    loc_dict=location_dict
                )
                acc_id = dict()
        if acc_id:
            executor.submit(
                accession_get_gene,
                acc_data=dict(acc_id),
                dest_dict=genes_by_id,
                loc_dict=location_dict
            )
    return genes_by_id


def table_add_gene_annotations(gene_table, location_table, origin):
    """
    Given a pandas dataframe expression table, searches NCBI for gene annotations
    for each GB (GeneBank) accession ID, using the file provided by course staff.
    rows without gene location will be removed from the table, the annotated table
    will be stored as an intermediate file, and if such an intremediate already exists,
    it will be returned instead of annotating the table from scratch.
    """
    filename = '{0}_annotated_expression.csv'.format(origin)
    if filename in os.listdir('./intermediate_files'):
        print("Intermediate annotated expression file found. Retrieving...")
        annotated_expression = pd.read_csv('./intermediate_files/' + filename, sep=',',
                                           dtype=str)
        return annotated_expression.set_index('ID')

    print("No intermediate annotated expression file. Annotating...")
    gene_loc_by_id = get_table_genes(gene_table, location_table.T.to_dict(orient='list'))
    negative_dummy_list = [np.nan for _ in range(6)]
    XY_handling = {'X': '23', 'Y': '24'}
    location_columnns = {
        'gene_name': [gene_loc_by_id.get(idv, negative_dummy_list)[0]
                      for _, idv in gene_table['ID'].iteritems()],
        'chromosome': [XY_handling.get(gene_loc_by_id.get(idv, negative_dummy_list)[3],
                                       gene_loc_by_id.get(idv, negative_dummy_list)[3])
                       for _, idv in gene_table['ID'].iteritems()],
        'start': [gene_loc_by_id.get(idv, negative_dummy_list)[4]
                  for i_, idv in gene_table['ID'].iteritems()],
        'end': [gene_loc_by_id.get(idv, negative_dummy_list)[5]
                for _, idv in gene_table['ID'].iteritems()]
    }

    gene_table = gene_table.assign(**location_columnns)
    gene_table.dropna(axis=0, how='any', inplace=True)
    gene_table.set_index('ID', inplace=True)
    gene_table.to_csv('./intermediate_files/' + filename)
    return gene_table


def table_remove_duplicate_genes(table, origin):
    filename = "{0}_no_duplicates.csv".format(origin)
    if filename in os.listdir('./intermediate_files'):
        print("Intermediate no duplicate expression file found. Retrieving...")
        annotated_expression = pd.read_csv('./intermediate_files/' + filename)
        return annotated_expression.set_index('ID')

    print("No intermediate no duplicate expression file. Removing duplicates...")
    print(set(table['chromosome']))
    """
    column_dtypes = {col: 'object' if col in {'gene_name', 'GB_ACC'} else 'float64' for col in table.columns}
    print(column_dtypes)
    table.astype(column_dtypes, copy=False, errors='raise')
    print(table.groupby(['gene_name', 'GB_ACC']).mean())
    """


if __name__ == "__main__":
    gene_locations = build_gene_location_dict()
    kidney_gse = geo.get_GEO(geo='GSE8356', destdir='./expression_data')
    kidney_table = generate_raw_expression_table(kidney_gse)
    blood_gse = geo.get_GEO(geo='GSE18067', destdir='./expression_data')
    blood_table = generate_raw_expression_table(blood_gse)

    idney_table = table_add_gene_annotations(kidney_table, gene_locations, 'kidney')
    blood_table = table_add_gene_annotations(blood_table, gene_locations, 'blood')
    table_remove_duplicate_genes(blood_table, 'blood')
