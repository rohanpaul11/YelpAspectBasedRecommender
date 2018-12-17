from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
import json
from server import *
import ast
from collections import defaultdict

CLEAR_LINE = '\033[K'


def get_sentences(text, parser):
    output = parser.api_call(
        text,
        properties={
            'annotators': 'tokenize,ssplit',
            'ssplit.boundaryTokenRegex': '[.!?;]+'
        })

    sentences = []
    # print(text)
    for sentence in output['sentences']:
        start_offset = sentence['tokens'][0]['characterOffsetBegin']
        end_offset = sentence['tokens'][-1]['characterOffsetEnd']
        sentence_str = text[start_offset:end_offset]
        # print(sentence_str,'\n')
        sentences.append(sentence_str)

    return sentences


def get_dependencies(review_file, dependency_file):
    parser = CoreNLPParser(url='http://localhost:9000')
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    with open(review_file, 'r') as fin, open(dependency_file, 'w') as fout:
        count = 0
        for line in fin:
            review = json.loads(line)
            review_text = review['text']

            print('Parsing review', str(count + 1), '....', end='\r')

            try:
                sentences = get_sentences(review_text, parser)
                review_dependencies = []
                for sentence in sentences:
                    # print(sentence)
                    parses = dep_parser.parse(
                        list(parser.tokenize(sentence.lower())))
                    dependencies = [[
                        (governor, dep, dependent)
                        for governor, dep, dependent in parse.triples()
                    ] for parse in parses]

                    review_dependencies.append(dependencies)

                fout.write('Review {}:'.format(review['review_id']))
                fout.write(str(review_dependencies) + '\n')
                count += 1
            except Exception as e:
                stop_corenlp_server()
                start_corenlp_server()
                parser = CoreNLPParser(url='http://localhost:9000')
                dep_parser = CoreNLPDependencyParser(
                    url='http://localhost:9000')
                fout.write('Review {}:FAILED - {}'.format(
                    review['review_id'], str(e)))


# extracts aspects using dependency parser
def get_aspects(dependency_file, aspect_file):
    count = 0
    with open(dependency_file,'r') as fin, open(aspect_file, 'w') as fout:
    # , open('../Data/advmod.txt', 'w') as adv, open('../Data/cmpd.txt', 'w') as cmpd, open('../Data/dobj.txt', 'w') as dobj, open('../Data/amod.txt', 'w') as amod:
        for line in fin:
            print('Review {}'.format(str(count + 1)), end='\r')
            index = line.find(':')
            review_id = line[:index + 1]
            review_dependencies = ast.literal_eval(line[index + 1:])
            aspects = []
            for sentence_dependencies in review_dependencies:
                # print(type(sentence_dependencies))

                for dependency in sentence_dependencies[0]:
                    # print(dependency)
                    if dependency[1] == 'nsubj' and dependency[0][1].startswith('JJ') and dependency[2][1].startswith('NN'):
                    # if (dependency[0][1].startswith('JJ') and dependency[2][1].startswith('NN')) or (dependency[0][1].startswith('NN') and dependency[2][1].startswith('JJ')):
                    # if (dependency[1] == 'nsubj' and
                    #     ((dependency[0][1].startswith('JJ')
                    #       and dependency[2][1].startswith('NN')) or
                    #      (dependency[0][1].startswith('NN')
                    #       and dependency[2][1].startswith('JJ')))) or (
                    #           dependency[1] == 'dobj' and
                    #            ((dependency[0][1].startswith('NN')
                    #             or dependency[0][1].startswith('VBN'))
                    #            and dependency[2][1].startswith('NN'))) or (
                    #           dependency[1] == 'advmod' and
                    #            (dependency[0][1].startswith('JJ')
                    #             or dependency[0][1].startswith('VBN'))
                    #            and dependency[2][1].startswith('RB')) or (
                    #           dependency[1] == 'compound'
                    #            and dependency[0][1].startswith('NN')
                    #            and dependency[2][1].startswith('NN')) or (
                    #           dependency[1] == 'amod'
                    #            and dependency[0][1].startswith('NN')
                    #            and dependency[2][1].startswith('JJ')):
                        aspects.append(dependency)
                        # if dependency[1] == 'dobj':
                        #     dobj.write(str(dependency) + '\n')
                        # if dependency[1] == 'advmod':
                        #     adv.write(str(dependency) + '\n')
                        # if dependency[1] == 'compound':
                        #     cmpd.write(str(dependency) + '\n')
                        # if dependency[1] == 'amod':
                        #     amod.write(str(dependency) + '\n')
            fout.write(review_id + str(aspects) + '\n')
            count += 1

def combine_dependencies(dependency_file, combined_dependency_file):
    with open(dependency_file, 'r') as fin, open(combined_dependency_file, 'wt') as fout:
        for line in fin:
            parts = line.strip().split(':', 1)
            review_id = parts[0]
            dependencies = ast.literal_eval(parts[1].strip())
            
            fout.write(review_id + '\n')            
            for sent_idx, sent_dep in enumerate(dependencies):                
                combined_deps = defaultdict(list)
                for dep in sent_dep[0]:
                    # print(dep)
                    head = dep[0]
                    dependent = (dep[2], dep[1])
                    combined_deps[head].append(dependent)
                fout.write("Sentence {}:\n".format(sent_idx))
                for word, deps in combined_deps.items():
                    fout.write('{}: {}\n'.format(word, str(deps)))
                
            fout.write('='*80 + '\n')
     

if __name__ == '__main__':
    review_file = '../Data/filtered_dataset.json'
    dependency_file = '../Data/dependencies.txt'
    aspect_file = '../Data/aspects.txt'
    combined_dependency_file = '../Data/combined_dependencies.txt'

    # combine_dependencies(dependency_file, combined_dependency_file)
# Runs dependency parser on train set to extract dependencies
    # retcode = start_corenlp_server()
    # if retcode != 0:
    #     exit(retcode)

    # get_dependencies(review_file, dependency_file)

    # retcode = stop_corenlp_server()
    # if retcode != 0:
    #     print('Failed to shutdown server properly!Please check and shut it down.')

# Extracts aspects from dependencies of train set
    # get_aspects(dependency_file, aspect_file)

    test_review_file = '../Data/negative_test_reviews.json'
    test_dependency_file = '../Data/test_dependencies.txt'
    test_aspect_file = '../Data/test_aspects.txt'

# Runs dependency parser on test set to extract dependencies
    # retcode = start_corenlp_server()
    # if retcode != 0:
    #     exit(retcode)

    # get_dependencies(test_review_file, test_dependency_file)

    # retcode = stop_corenlp_server()
    # if retcode != 0:
    #     print('Failed to shutdown server properly!Please check and shut it down.')

# Extracts aspects from dependencies of test set
    # get_aspects(test_dependency_file, test_aspect_file)