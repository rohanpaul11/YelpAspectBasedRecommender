from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.corenlp import CoreNLPParser
import json
from server import *
import ast

CLEAR_LINE = '\033[K'

def get_sentences(text, parser):
    output = parser.api_call(text, properties={'annotators': 'tokenize,ssplit', 'ssplit.boundaryTokenRegex':'[.!?;]+'})
    
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
                        
            print('Parsing review', str(count + 1),'....',end='\r')            

            try:
                sentences = get_sentences(review_text, parser)   
                review_dependencies = []
                for sentence in sentences:
                    # print(sentence)            
                    parses = dep_parser.parse(list(parser.tokenize(sentence.lower())))
                    dependencies = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]

                    review_dependencies.append(dependencies)
                
                fout.write('Review {}:'.format(review['review_id']))
                fout.write(str(review_dependencies) + '\n')
                count += 1    
            except Exception as e:
                stop_corenlp_server()
                start_corenlp_server()
                parser = CoreNLPParser(url='http://localhost:9000')
                dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
                fout.write('Review {}:FAILED - {}'.format(review['review_id'], str(e)))

# incomplete
def get_aspects(dependency_file, aspect_file):
    count = 0
    with open(dependency_file, 'r') as fin, open(aspect_file, 'w') as fout:                
        for line in fin:
            print('Review {}'.format(str(count+1)), end='\r')
            index = line.find(':')
            review_id = line[:index+1]
            review_dependencies = ast.literal_eval(line[index + 1:])
            aspects = []
            for sentence_dependencies in review_dependencies:
                # print(type(sentence_dependencies))                                
                for dependency in sentence_dependencies[0]:
                    # print(dependency)
                    if dependency[1] == 'nsubj' and dependency[0][1].startswith('JJ') and dependency[2][1].startswith('NN'):
                        aspects.append(dependency)
            fout.write(review_id + str(aspects) + '\n')
            count += 1

if __name__ == '__main__':
    review_file = '../Data/filtered_dataset.json'
    dependency_file = '../Data/dependencies.txt'
    aspect_file = '../Data/aspects.txt'

    retcode = start_corenlp_server()
    if retcode != 0:
        exit(retcode)
    
    get_dependencies(review_file, dependency_file)

    retcode = stop_corenlp_server()
    if retcode != 0:
        print('Failed to shutdown server properly!Please check and shut it down.')
    
    get_aspects(dependency_file, aspect_file)