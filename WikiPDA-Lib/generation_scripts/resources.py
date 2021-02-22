"""
Module is used to generate all necessary key-value store data fro library, but assumes a rather
specific folder structure on the input and output.
"""
import argparse
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
from pyspark.mllib.recommendation import MatrixFactorizationModel
import plyvel
import pickle
import os
from tqdm import tqdm
import numpy as np
import tarfile
import os.path
import shutil
import re


def write_db_dictionary(path, dictionary, key_encoding=str.encode, value_encoding=str.encode, batch_size=1000000):
    """
    Helper function which writes to a plyvel database in batches using a generator function.
    Also writes the given file as a dictionary at the same time to save time.
    :param db_path: Path on disk to the plyvel database.
    :param dictionary: The dictionary to dump.
    :param key_encoding: The function to call on the key to create a Bytes object.
    :param value_encoding: The function to call on the value to create a Bytes object.
    :param batch_size: How many items to write at once to the database.
    """
    db = plyvel.DB(path + '.db', create_if_missing=True)
    with db.write_batch() as wb:
        i = 0
        for key, value in dictionary.items():

            wb.put(key_encoding(key), value_encoding(value))

            # Poor man's modulo n
            i += 1
            if i == batch_size:
                wb.write()
                i = 0

        # Write last remaining batch
        wb.write()
    db.close()

    # Now also dump the pickled version of the dictionary itself
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(dictionary, f)



def make_tarfile(output_filename: str, source_dir: str):
    """
    Compresses the given directory into a tarball (*.tar.gz) and deletes the original directory.
    :param output_filename: The name and path of the tarball.
    :param source_dir: The path to the directory which should be compressed.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    # Remove old directory
    shutil.rmtree(source_dir)


def dump_full_matrices(spark: SparkSession, matrix_factorization_path: str, base_path: str):
    """
    Dumps the matrix decomposition of a given pyspark MatrixFactorizationModel instance.
    Dumps have the names 'users.npy' and 'products.npy' and are numpy matrices.
    :param spark: SparkSession object to load and dump the matrices using
    :param matrix_factorization_path: Path to the MatrixFactorizationModel on disk
    :param base_path: The directory to dump the resulting files in (as 'users.npy' and
    'products.npy')
    """

    # Load model
    matrix_factorization = MatrixFactorizationModel.load(spark, path=matrix_factorization_path)

    # Start saving users
    users = matrix_factorization.userFeatures()
    indices = matrix_factorization.userFeatures().map(lambda x: x[0])
    u = np.zeros((indices.max() + 1, 300))

    for index, vector in tqdm(users.collect()):
        u[index] = np.array(vector)

    np.save(base_path + 'users.npy', u)
    del u

    # Now save products
    products = matrix_factorization.productFeatures()
    indices = matrix_factorization.productFeatures().map(lambda x: x[0])
    v = np.zeros((indices.max() + 1, 300))

    for index, vector in tqdm(products.collect()):
        v[index] = np.array(vector)

    np.save(base_path + 'products.npy', v)
    del v


def create_anchors(spark: SparkSession, parquet_path: str, beta_path: str, output_path: str):
    """
    Dumps the n-gram to QID mappings so that link densification can be done later.
    Dumps both a pickled dictionary and a plyvel database.
    :param spark: SparkSession object to load and dump the matrices using
    :param parquet_path: Path to the parquet file containing the mappings
    :param beta_path: Path to the parquet file containing the beta coefficients
    :param output_path: The path to dump the mappings
    """
    anchors = spark.read.parquet(parquet_path)
    beta = spark.read.parquet(beta_path)
    beta_filter = beta.where("as_link >= 10 and beta >= 0.065").select("anchor")
    filtered_anchors = anchors.join(beta_filter, 'anchor')
    candidates = filtered_anchors.filter("anchor not rlike '^[0-9]+$'").where("LENGTH(anchor)>0") \
        .groupBy("anchor").agg(collect_set("destination_qid").alias("candidates"))
    anchor_mapping = {row.anchor:row.candidates for row in candidates.rdd.toLocalIterator()}

    # Dump mapping
    def encode_as_list(elements):

        # To make sure we can still decode as list
        if len(elements) < 2:
            return str.encode(elements[0] + ';')
        return str.encode(';'.join(elements))
    write_db_dictionary(output_path, anchor_mapping, value_encoding=encode_as_list)
    del anchor_mapping


def dump_matrix_positions(spark: SparkSession, matrix_positions_path: str, out_path: str):
    """
    Dumps the mappings between QIDs and indices in the matrix decomposition.
    :param spark: SparkSession object to load and dump the matrices using
    :param matrix_positions_path: Path to the parquet file containing the mappings
    :param out_path: The path to dump the mappings
    :return:
    """
    matrix_positions = spark.read.parquet(matrix_positions_path)
    positions_mapping = {qid: matrix_index
                         for matrix_index, qid in matrix_positions.rdd.toLocalIterator()}

    # Dump mapping
    write_db_dictionary(out_path, positions_mapping, value_encoding=lambda x: str.encode(str(x)))
    del positions_mapping


def extract_title_qid_mapping(spark: SparkSession, links_path: str, language: str, out_path: str):
    """
    Dump the mappings between article titles and their QIDs.
    Dumps both a pickled dictionary and a plyvel database.
    :param spark: SparkSession object to load and dump the matrices using
    :param links_path: Path to the parquet file containing (among other things) the mappings
    :param language: Language to use in filter to only keep links for a given language edition
    :param out_path: The path to dump the mappings
    """
    lang_links = spark.read.parquet(links_path)
    filtered = lang_links.filter(lang_links.site == language)
    title_qid_mapping = {row.title: row.qid for row in filtered.rdd.toLocalIterator()}

    # Dump dictionary
    write_db_dictionary(out_path, title_qid_mapping)
    del title_qid_mapping


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate the language specific data for WikiPDA.')
    parser.add_argument('--input', help='input resource folder', required=True)
    args = parser.parse_args()

    conf = pyspark.SparkConf().setMaster("local[12]").setAll([
                                        ('spark.driver.memory', '20g'),
                                        ('spark.driver.maxResultSize', '32G'),
                                        ('spark.local.dir', '/scratch/tmp/'),
                                        ('spark.yarn.stagingDir', '/scratch/tmp/'),
                                        ('spark.sql.warehouse.dir', '/scratch/tmp/')
    ])

    # Create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext

    # Iterate over each supported language and generate resources
    resources_path = f'../resources/'
    for lang in os.listdir(args.input):

        # Skip if not resource folder
        lang_regex = r'\w*wiki'
        if not re.match(lang_regex, lang) or re.search(lang_regex, lang).group(0) != lang:
            continue

        # Also skip if already processed
        output_file = resources_path + lang + '.tar.gz'
        if os.path.isfile(output_file):
            print(f'{lang} Already processed, skipping...')
            continue

        print(f'Processing resources for {lang}...')
        lang_res = resources_path + lang + '/'
        input_path = f'{args.input}/{lang}/'
        os.mkdir(lang_res)

        create_anchors(spark, input_path + 'anchors_info_qid.parquet',
                       input_path + 'anchors_beta.parquet',
                       lang_res + 'qid_mappings')

        dump_matrix_positions(spark, input_path + 'matrix_positions.parquet',
                              lang_res + 'matrix_positions')

        dump_full_matrices(spark, input_path + 'ALSModel.model', lang_res)

        extract_title_qid_mapping(spark, args.input + '/WikidataInterlanguageLinks.parquet', lang,
                                  lang_res + 'title_qid_mappings')

        # Compress results to conserve space and prepare for distribution
        make_tarfile(output_file, lang_res)
