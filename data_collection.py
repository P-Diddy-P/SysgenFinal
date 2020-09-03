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
                insert_name = str(gsm_name)
                if gsm_name in name_table:
                    insert_name += ('.' + str(name_table[gsm_name]))

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

    NOTE: GET YOUR OWN DAMN ENTREZ API KEY FOR THIS.
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
    xy_handling = {'X': '23', 'Y': '24'}
    location_columnns = {
        'gene_name': [gene_loc_by_id.get(idv, negative_dummy_list)[0]
                      for _, idv in gene_table['ID'].iteritems()],
        'chromosome': [xy_handling.get(gene_loc_by_id.get(idv, negative_dummy_list)[3],
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
    if origin:
        gene_table.to_csv('./intermediate_files/' + filename)
    return gene_table


def table_remove_duplicate_genes(table):
    """
    Given a table, removes all rows of expression for the same gene,
    and replaces them with a single row for each gene, with the mean expression of all rows.
    Since by this time IDs of individual genes are not used, the first ID for each gene name
    is used as a convenient default.
    """
    mean_rows = dict()
    mean_genes = set()
    for row_id, dup in table.duplicated('gene_name', keep=False).iteritems():
        if dup and table['gene_name'][row_id] not in mean_genes:
            mean_genes.add(table['gene_name'][row_id])

            row = table.loc[row_id]
            dup_rows = table.loc[table['gene_name'] == table.loc[row_id]['gene_name']]
            dup_expression = dup_rows[[col for col in table.columns if col.startswith('BXD')]]
            mean_expression = dup_expression.astype('float64').mean(axis=0, numeric_only=True)
            mean_rows[row_id] = list(row[['GB_ACC', 'gene_name', 'chromosome', 'start', 'end']]) \
                + list(mean_expression)

    table.drop_duplicates('gene_name', keep=False, inplace=True)
    mean_dataframe = pd.DataFrame.from_dict(mean_rows, orient='index', columns=list(table.columns))
    return pd.concat([table, mean_dataframe])


def table_remove_duplicate_strains(table):
    """
    Given a table, remove all columns starting with the same BXD strain and inserts
    columns with a mean of all individuals of the same strain.
    """
    def get_bxd_number(s):
        res = s[0:4]
        if len(s) >= 5 and s[4] in {str(i) for i in range(10)}:
            res += s[4]
        return res

    table_expression = table[[col for col in table.columns if col.startswith('BXD')]].astype('float64')
    table_grouped = table_expression.groupby(by=get_bxd_number, axis=1, as_index=True).mean()
    table_identification = table[[col for col in table.columns if not col.startswith('BXD')]]
    return table_identification.merge(table_grouped, how='inner', left_index=True, right_index=True)


def table_remove_duplicates(table, origin):
    """
    Given an annotated expression table, removes duplicate rows (multiple tests for the same gene in each strain)
    and duplicate columns (multiple individuals of each strain). The resulting table is stored
    as an intermediate file. If such a file already exists, return it instead.
    """
    filename = "{0}_no_duplicates.csv".format(origin)
    if filename in os.listdir('./intermediate_files'):
        print("Intermediate expression file with no duplicates found. Retrieving...")
        no_duplicate_expression = pd.read_csv('./intermediate_files/' + filename)
        return no_duplicate_expression.set_index('ID')

    print("No intermediate no duplicate expression file. Removing duplicates...")
    no_duplicate_strains = table_remove_duplicate_strains(table)
    no_duplicates = table_remove_duplicate_genes(no_duplicate_strains)
    if origin:
        no_duplicates.to_csv('./intermediate_files/' + filename)
    return no_duplicates


def tables_drop_unique_strains(table1, table2):
    """
    Given 2 gene expression tables, drops all columns with strains not shared by
    both tables.
    """
    shared_columns = set(table1.columns) & set(table2.columns)
    table1_dropped = table1.drop(columns=[col for col in table1.columns if col not in shared_columns])
    table2_dropped = table2.drop(columns=[col for col in table2.columns if col not in shared_columns])

    return table1_dropped, table2_dropped


def table_filter(table, by_var=True, vl=1.25, vu=2.0, by_mean=False, ml=-1.0, mu=1.0):
    """
    Filters a given table by argument constraints on the mean and variance of
    gene expression distribution among different samples (i.e. the mean and variance of
    each row). The filtered table should contain about 1000 genes at most.
    """
    expression_table = table[[col for col in table.columns if col.startswith('BXD')]].astype('float64')
    row_variance = expression_table.var(axis=1, numeric_only=True)
    row_mean = expression_table.mean(axis=1, numeric_only=True)

    mv = row_variance.mean()
    mm = row_mean.mean()

    filtered_table = table.loc[[row_id for row_id, _ in expression_table.iterrows() if
                                (not by_var or vl*mv < row_variance[row_id] < vu*mv) and
                                (not by_mean or ml*mm < row_mean[row_id] < mu*mm)]]
    return filtered_table


def gene_expression_pipeline(geo_id, tissue_origin, gene_locations):
    """
    Given a GEO id and a gene location table, runs the whole data
    pipeline of annotation -> duplication removal -> filtering on the GEO gene expression
    table.
    The tissue origin parameter is given to save intermediate results of the pipeline
    and reduce runtime when trying different filtering parameters. An empty string can be
    supplied to disallow saving of intermediate files.

    NOTE:
    1. GEOparse will download an expression SOFT file and save it in destdir unless one
        is already provided.
    2. table_add_gene_annotations requires several minutes to run (because of eutils requests).
       Do NOT terminate early.
    3. After running table_remove_duplicates, for some reason the ID column header is not saved
       in the csv. This can be easily fixed manually once (not worth code intervention).
    """
    gse_data = geo.get_GEO(geo=geo_id, destdir='./expression_data')
    expression_table = generate_raw_expression_table(gse_data)
    expression_table = table_add_gene_annotations(expression_table, gene_locations, tissue_origin)
    expression_table = table_remove_duplicates(expression_table, tissue_origin)
    filtered_expression = table_filter(expression_table)

    return filtered_expression


if __name__ == "__main__":
    gene_locations = build_gene_location_dict()
    liver_table = gene_expression_pipeline('GSE17522', 'liver', gene_locations)
    kidney_table = gene_expression_pipeline('GSE8356', 'kidney', gene_locations)

    liver_table, kidney_table = tables_drop_unique_strains(liver_table, kidney_table)
