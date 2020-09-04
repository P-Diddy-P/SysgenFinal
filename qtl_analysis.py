from scipy import stats
import pandas as pd

ALLELIC_VALUES = {'B': 2, 'H': 1, 'D': 0, 'U': -1,
                  'b': 2, 'h': 1, 'd': 0, 'u': -1}


def get_regression_coefficients(xy):
    avg_x = sum([x for x, y in xy]) / len(xy)
    avg_y = sum([y for x, y in xy]) / len(xy)

    b1 = sum([(x - avg_x) * (y - avg_y) for x, y in xy]) / \
        sum([(x - avg_x)**2 for x, y in xy])
    b0 = avg_y - avg_x * b1

    return b0, b1


def variance_ratio(xy, b0, b1):
    avg_y = sum([y for x, y in xy]) / len(xy)
    est_y = [b0 + x * b1 for x, y in xy]

    sse = sum([(xy[i][1] - est_y[i])**2 for i in range(len(xy))])
    ssr = sum([(y - avg_y)**2 for y in est_y])

    return ssr * (len(xy) - 2) / sse


def snp_regression(genotype, phenotype):
    reg_xy = []
    for strain, allele in genotype.iteritems():
        allele_value = ALLELIC_VALUES.get(allele, -1)
        if "BXD" in strain and pd.notna(phenotype[strain]) and allele_value >= 0:
            reg_xy.append((allele_value, phenotype[strain]))

    # There is no regression to perform if all x values are the same (no allele variation)
    # or there are less then 3 points to regress on (no estimation in the regression)
    assert len(set([x for x, y in reg_xy])) > 1 and len(reg_xy) > 2
    b0, b1 = get_regression_coefficients(reg_xy)
    f_star = variance_ratio(reg_xy, b0, b1)

    return stats.f.sf(f_star, 1, len(reg_xy) - 2)


def parse_snp_locations(snp_iterator):
    last_chromosome_snp = {}

    for snp in snp_iterator:
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
    chromosome = 1
    total = 0
    starts = [0]
    while chromosome in chromosome_sizes:
        total += chromosome_sizes[chromosome]
        starts.append(total)
        chromosome += 1
    return starts
