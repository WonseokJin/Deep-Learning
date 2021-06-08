import time
import numpy as np
from scipy.special import gammaln
import re
import random


class LDAGibbs:
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003

    Latent Dirichlet allocation with collapsed Gibbs sampling
    """
    def __init__(self, num_topics, doc_file_path, vocas, output_dir_name, alpha=0.1, beta=0.01):
        """
        Constructor method
        Do not modify this function
        :param num_topics: the number of topics
        :param doc_file_path: BOW document file path
        :param vocas: vocabulary list
        :param output_dir_name: output directory name
        :param alpha: alpha value in LDA
        :param beta: beta value in LDA
        :return: void
        """
        self.docs = self.read_bow(doc_file_path)
        self.words = vocas
        self.K = num_topics
        self.D = len(self.docs)
        self.W = len(vocas)
        self.output_dir_name = output_dir_name

        # Hyper-parameters
        # We use symmetric hyper-parameters
        self.alpha = alpha
        self.beta = beta

        # self.WK: Words by Topics matrix,
        # self.DK: Documents by Topics matrix
        self.WK = np.zeros([self.W, self.K])
        self.DK = np.zeros([self.D, self.K])

        # Random initialization of topics
        np.random.seed(108)
        self.doc_topics = list()    # Topic index of each word in each document

        for di in range(self.D):
            doc = self.docs[di]
            topics = np.random.randint(self.K, size=len(doc))
            self.doc_topics.append(topics)

            for wi in range(len(doc)):
                topic = topics[wi]
                word = doc[wi]
                self.WK[word, topic] += 1
                self.DK[di, topic] += 1

    def run(self, max_iter=2000, do_print_log=False):
        """
        Run Collapsed Gibbs sampling for LDA
        Do not modify this function
        :param max_iter: Maximum number of gibbs sampling iteration
        :param do_print_log: Do print loglikelihood and run time
        :return: void
        """
        if do_print_log:
            prev = time.clock()
            for iteration in range(max_iter):
                print('\n이번 iter : ',iteration,' 시간 : ' ,time.clock() - prev, ' log : ',self.loglikelihood())
                prev = time.clock()
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self.export_result(output_file_name="iter_{}".format(iteration))
        else:
            for iteration in range(max_iter):
                self._gibbs_sampling()
                if 99 == iteration % 100:
                    self.export_result(output_file_name="iter_{}".format(iteration))

    def _gibbs_sampling(self):
        """
        Run Gibbs Sampling
        :return: void
        """
        # For each document
        for di in range(self.D):
            doc = self.docs[di]
            print('\r이번 문서 :', di,'/',self.D,end="")
            # For each word
            for wi in range(len(doc)):
                word = doc[wi]

                # 1. Decrement
                # Hint) use 'old_topic' variable
                old_topic = self.doc_topics[di][wi]
                self.WK[word, old_topic] -= 1
                self.DK[di, old_topic] -= 1

                # 2. Sample
                # Hint) use self._sampling_from_dist function
                vector=[]
                for topic in range(self.K):
                    P_DK = self.DK[di, topic]
                    P_WK = self.WK[word, topic]
                    vector.append(P_DK * P_WK)
                nanum = sum(self.DK[di, :]) * sum(self.WK[:, topic])
                vector = [x / nanum for x in vector]
                #for topic in range(self.K):
                #    P_DK = self.DK[di, topic]
                #    vector.append(P_DK)
                #nanum=sum(self.DK[di,:])
                #vector = [x / nanum for x in vector]
                """ 6분 1500,0000 (WK계산이 존나 느림) -> DK만 쓰는게 훨씬 빠르고 결과도 비슷!
                for topic in range(self.K):
                    P_DK=self.DK[di,topic]
                    vector.append(P_DK)
                nanum=sum(self.DK[di,:])
                vector=[x/nanum for x in vector]
                """
                """ 3시간 30분 1800,0000
                for topic in range(self.K):
                    P_DK=self.DK[di,topic]
                    P_WK=self.WK[word,topic]
                    vector.append(P_DK*P_WK)
                nanum=sum(self.DK[di,:])*sum(self.WK[:,topic])
                vector=[x/nanum for x in vector]
                """
                #P_DK=self.DK[di,:]/sum(self.DK[di,:])
                #P_WK=self.WK[word,:]/sum(self.WK[word,:])
                #print('P_DK : ',(P_DK))
                #print('P_WK : ',(P_WK))
                #vector = [P_DK*P_WK for P_DK,P_WK in zip(P_DK,P_WK)]
                vector=np.array(vector)
                #choose=random.choice(vector)
                #new_topic=vector.index(choose)
                new_topic = self._sampling_from_dist(vector)
                #new_topic = self._sampling_from_dist(self.WK[word,:]/sum(self.WK[word,:]))

                # 3. Increment
                # Hint) use 'new_topic' variable
                self.WK[word, new_topic] += 1
                self.DK[di, new_topic] += 1

                pass

    @staticmethod
    def _sampling_from_dist(prob_vec):
        """
        Multinomial sampling with probability vector
        Do not modify this function
        :param prob_vec: probability vector
        :return: a new sample, it is new topic index
        """
        thr = prob_vec.sum() * np.random.rand()
        new_topic = 0
        tmp = prob_vec[new_topic]
        while tmp < thr :
            new_topic += 1
            if new_topic == 50 :
                new_topic=49
                break
            tmp += prob_vec[new_topic]
        return new_topic

    def loglikelihood(self):
        """
        Compute log likelihood function
        Do not modify this function
        :return: log likelihood function
        """
        return self._topic_loglikelihood() + self._document_loglikelihood()

    def _topic_loglikelihood(self):
        """
        Compute log likelihood by topics
        Do not modify this function
        :return: log likelihood by topics
        """
        ll = self.K * gammaln(self.beta * self.W)
        ll -= self.K * self.W * gammaln(self.beta)

        for ki in range(self.K):
            ll += gammaln(self.WK[:, ki] + self.beta).sum() - gammaln(self.WK[:, ki].sum() + self.beta)

        return ll

    def _document_loglikelihood(self):
        """
        Compute log likelihood by documents
        Do not modify this function
        :return: log likelihood by documents
        """
        ll = self.D * gammaln(self.alpha * self.K)
        ll -= self.D * self.K * gammaln(self.alpha)

        for di in range(self.D):
            ll += gammaln(self.DK[di, :] + self.alpha).sum() - gammaln(self.DK[di, :].sum() + self.alpha)

        return ll

    def export_result(self, output_file_name, rank_idx=10):
        """
        Export Algorithm Result to File
        Do not modify this function
        :param output_file_name: output file name
        :param rank_idx:
        :return: the number of printed words in a topic in output file
        """
        # Raw data
        np.savetxt("{}/WK_{}.csv".format(self.output_dir_name, output_file_name), self.WK, delimiter=",")
        np.savetxt("{}/DK_{}.csv".format(self.output_dir_name, output_file_name), self.DK, delimiter=",")

        # Ranked words in topics
        with open("{}/Topic_topwords_{}.csv".format(self.output_dir_name, output_file_name), "w") as ranked_topic_word_file:
            topic_idx = -1
            for topic_vec in self.WK.T:
                topic_idx += 1
                sorted_words = sorted(enumerate(topic_vec), key=lambda x: x[1], reverse=True)
                print('topic {},{}'.format(topic_idx,
                                           ",".join([self.words[x[0]] for x in sorted_words[:rank_idx]])),
                      file=ranked_topic_word_file)

    @staticmethod
    def read_bow(file_path):
        """
        Read BOW file to run topic models with Gibbs sampling
        Do not modify this function
        :param file_path: The path of BOW file
        :return: documents list
        """
        split_pattern = re.compile(r'[ :]')
        docs = list()

        with open(file_path, 'r') as bow_file:
            for each_line in bow_file:
                split_line = split_pattern.split(each_line)
                cur_doc = list()
                word_ids = [int(x) for x in split_line[2::2]]
                word_counts = [int(x) for x in split_line[3::2]]

                for word_id, word_ct in zip(word_ids, word_counts):
                    for each_time in range(word_ct):
                        cur_doc.append(word_id)

                docs.append(cur_doc)

        return docs
