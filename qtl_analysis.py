from sys import argv

from scipy import stats
from numpy import log10
import pandas as pd
import pylab

PHENOTYPE = "Amygdala, basolateral complex (LaDL, LaVL, LaVM, BLP, and BLA), glial cell density [n/mm^3]"
GV = {'B': 2, 'D': 0, 'U': -1, 'H': 1,
      'b': 2, 'd': 0, 'u': -1, 'h': 1}
PLOT = False

"""
    Note: this program uses Pandas native functions in order to interface with excel.
    As such, additional dependancies required are: xlrd and openpyxl
"""


def process_arguments():
    genotype_data = pd.read_excel(argv[1], header=0, index_col=0, skiprows=[0])
    phenotype_data = pd.read_excel(argv[2], header=0, index_col=0)

    return phenotype_data, genotype_data


def get_regression_coefficients(xy):
    avg_x = sum([e[0] for e in xy]) / len(xy)
    avg_y = sum([e[1] for e in xy]) / len(xy)

    b1 = sum([(e[0] - avg_x) * (e[1] - avg_y) for e in xy]) / sum([(e[0] - avg_x)**2 for e in xy])
    b0 = avg_y - avg_x * b1

    return b0, b1


def variance_ratio(xy, b0, b1):
    avg_y = sum([e[1] for e in xy]) / len(xy)
    est_y = [b0 + e[0] * b1 for e in xy]

    sse = sum([(xy[i][1] - est_y[i])**2 for i in range(len(xy))])
    ssr = sum([(e - avg_y)**2 for e in est_y])

    return ssr * (len(xy) - 2) / sse


def snp_regression(row, phenotype):
    reg_xy = []
    for line in phenotype.iteritems():
        if "BXD" in line[0] and pd.notna(line[1]) and GV[row[line[0]]] >= 0:
            reg_xy.append((GV[row[line[0]]], line[1]))

    b0, b1 = get_regression_coefficients(reg_xy)
    f_star = variance_ratio(reg_xy, b0, b1)

    return -log10(stats.f.sf(f_star, 1, len(reg_xy) - 2))


def parse_snp_locations(snpiterator):
    last_chromosome_snp = {}

    for snp in snpiterator:
        snp_chr = snp[1]["Chr_Build37"]
        snp_loc = snp[1]["Build37_position"]

        if snp_loc > last_chromosome_snp.get(snp_chr, 0):
            last_chromosome_snp[snp_chr] = snp_loc

    mean_length = sum([e[1] for e in last_chromosome_snp.items()]) / len(last_chromosome_snp)

    return last_chromosome_snp, {e[0]: e[1]/mean_length for e in last_chromosome_snp.items()}


def extract_location(snp_data, chr_lengths, chr_sizes):
    snp_chr = snp_data["Chr_Build37"]
    snp_loc = snp_data["Build37_position"]
    x_loc = 0

    for i in range(1, snp_chr):
        x_loc += chr_sizes[i]
    x_loc += (snp_loc / chr_lengths[snp_chr]) * chr_sizes[snp_chr]
    return x_loc


def chromosome_starts(chromosome_sizes):
    chr = 1
    total = 0
    starts = [0]
    while chr in chromosome_sizes:
        total += chromosome_sizes[chr]
        starts.append(total)
        chr += 1
    return starts


if __name__ == "__main__":
    phenotypes, genotypes = process_arguments()
    snp_result = [(snp, snp_regression(snp_data, phenotypes.loc[PHENOTYPE])) for
                  snp, snp_data in genotypes.iterrows()]

    if PLOT:
        max_r = 0
        max_i = 0
        for i in range(len(snp_result)):
            if snp_result[i][1] > max_r:
                max_i = i
                max_r = snp_result[i][1]
        print(snp_result[max_i])

        chromosome_lengths, normalized_sizes = parse_snp_locations(genotypes.iterrows())
        snp_locations = [extract_location(snp[1], chromosome_lengths, normalized_sizes)
                         for snp in genotypes.iterrows()]

        pylab.scatter(snp_locations, [res[1] for res in snp_result])
        pylab.xlabel("normalized location on genome")
        pylab.ylabel("-log(p_value)")
        pylab.xticks(ticks=chromosome_starts(normalized_sizes), labels=normalized_sizes.keys())
        pylab.grid(True, which='major', axis='x')
        pylab.show()

    else:
        xldf = pd.DataFrame({
            "snp": [snp[0] for snp in snp_result],
            "-log p_value": [snp[1] for snp in snp_result]
        })
        xldf.to_excel("Analysis.xlsx", index=False)
