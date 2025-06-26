import numpy as np

from algorithms.ClusteringTemplate import Clustering


class Clope(Clustering):

    def __init__(self, r):
        super().__init__()
        self.r = r
        self.iters = None

    @staticmethod
    def update_supp(t: list, supp: dict, delete: bool = False) -> dict:
        """

        :param t:
        :param supp:
        :param delete:
        :return:
        """
        if delete is False:
            for i in t:
                if i in supp.keys():
                    supp[i] += 1
                else:
                    supp[i] = 1
        else:
            for i in t:
                if supp[i] == 1:
                    del supp[i]
                else:
                    supp[i] -= 1
        return supp

    def clope_insert(self, df: list, r: float) -> (list, dict, dict, dict):
        """

        :param df:
        :param r:
        :return:
        """
        C = [0]
        clusters = {0: {i: 1 for i in df[0]}}
        C_sizes = {0: 1}
        C_sum = {0: len(df[0])}

        for t in df[1:]:
            len_t = len(t)
            max_profit = 0
            best_cluster = 0
            for cluster, supp in clusters.items():
                new_profit = self.update_profit(t, len_t, supp, C_sizes[cluster], C_sum[cluster], r)
                if new_profit > max_profit:
                    max_profit = new_profit
                    best_cluster = cluster

            if max_profit < len_t / (len_t ** r):
                new_index = max(clusters.keys()) + 1
                C.append(new_index)
                clusters[new_index] = {i: 1 for i in t}
                C_sizes[new_index] = 1
                C_sum[new_index] = len_t
            else:
                C.append(best_cluster)
                clusters[best_cluster] = self.update_supp(t, clusters[best_cluster])
                C_sizes[best_cluster] += 1
                C_sum[best_cluster] += len_t

        return C, clusters, C_sizes, C_sum

    @staticmethod
    def rename_keys(clusters: dict, C_sizes: dict, C_sum: dict, C: list) -> (dict, dict, dict, list):
        """

        :param clusters:
        :param C_sizes:
        :param C_sum:
        :param C:
        :return:
        """
        sorted_keys = sorted(C_sizes.keys())
        new_clusters, new_C_sizes, new_C_sum = {}, {}, {}
        new_C = [None] * len(C)

        for i, key in enumerate(sorted_keys):
            new_clusters[i] = clusters[key]
            new_C_sizes[i] = C_sizes[key]
            new_C_sum[i] = C_sum[key]
            for k in range(len(C)):
                if C[k] == key:
                    new_C[k] = i

        return new_clusters, new_C_sizes, new_C_sum, new_C

    def clope_move(self, df: list, r: float, C: list, clusters: dict, C_sizes: dict, C_sum: dict, full_output: bool = False)\
            -> (list, int):
        """

        :param df:
        :param r:
        :param C:
        :param clusters:
        :param C_sizes:
        :param C_sum:
        :param full_output:
        :return:
        """
        iters = 0
        move = True
        while move:
            if iters >= 10:
                break
            move = False
            for i, t in enumerate(df):
                c_num = C[i]
                cluster_supp = clusters[c_num]
                c_size = C_sizes[c_num]
                len_t = len(t)
                if c_size == 1:
                    drop_profit = len_t / (len_t ** r)
                else:
                    drop_profit = self.get_drop_profit(t, len_t, cluster_supp, c_size, C_sum[c_num], r)

                new_cluster, best_cluster, max_profit = self.find_best_cluster(clusters, C_sizes, C_sum, c_num, t, len_t, r)

                if max_profit - drop_profit > 0:
                    move = True
                    clusters, C_sizes, C_sum = self.move_object(clusters, C_sizes, C_sum, c_num, best_cluster, new_cluster, t,
                                                           len_t)
                    C[i] = best_cluster

            if move:
                clusters, C_sizes, C_sum, C = self.rename_keys(clusters, C_sizes, C_sum, C)

            iters += 1

            if full_output is True:
                return C, clusters, C_sizes, C_sum

        return C, iters

    def move_object(self, clusters: dict, C_sizes: dict, C_sum: dict , c_num: dict, best_cluster: int, new_cluster: int,
                    t: list, len_t: int) -> (dict, dict, dict):
        """

        :param clusters:
        :param C_sizes:
        :param C_sum:
        :param c_num:
        :param best_cluster:
        :param new_cluster:
        :param t:
        :param len_t:
        :return:
        """
        if C_sizes[c_num] == 1:
            del clusters[c_num]
            del C_sizes[c_num]
            del C_sum[c_num]
        else:
            clusters[c_num] = self.update_supp(t, clusters[c_num], True)
            C_sizes[c_num] -= 1
            C_sum[c_num] -= len_t

        if new_cluster:
            clusters[best_cluster] = {j: 1 for j in t}
            C_sizes[best_cluster] = 1
            C_sum[best_cluster] = len_t

        else:
            clusters[best_cluster] = self.update_supp(t, clusters[best_cluster])
            C_sizes[best_cluster] += 1
            C_sum[best_cluster] += len_t

        return clusters, C_sizes, C_sum

    def find_best_cluster(self, clusters: dict, C_sizes: dict, C_sum: dict, c_num: int, t: list, len_t: int, r: float) \
            -> (int, int, float):
        """

        :param clusters:
        :param C_sizes:
        :param C_sum:
        :param c_num:
        :param t:
        :param len_t:
        :param r:
        :return:
        """
        max_profit = 0
        best_cluster = 0
        new_cluster = False
        for cluster, supp in clusters.items():
            if c_num != cluster:
                new_profit = self.update_profit(t, len_t, supp, C_sizes[cluster], C_sum[cluster], r)
                if new_profit > max_profit:
                    max_profit = new_profit
                    best_cluster = cluster

        new_profit = len_t / (len_t ** r)
        if max_profit < new_profit:
            new_cluster = True
            max_profit = new_profit
            best_cluster = max(clusters.keys()) + 1

        return new_cluster, best_cluster, max_profit

    @staticmethod
    def update_profit(t: list, len_t: int, supp: dict, old_size: int, old_s: int, r: float) -> float:
        """

        :param t:
        :param len_t:
        :param supp:
        :param old_size:
        :param old_s:
        :param r:
        :return:
        """
        old_w = len(supp)
        new_size = old_size + 1
        new_s = old_s + len_t
        new_w = old_w
        for item in t:
            if item not in supp.keys():
                new_w += 1

        if new_w == 0 or old_w == 0:
            print(new_s, new_size, new_w, old_s, old_size, old_w, supp)
        return (new_s * new_size) / (new_w ** r) - ((old_s * old_size) / (old_w ** r))

    @staticmethod
    def get_drop_profit(t: list, len_t: int, supp: dict, old_size: int, old_s: int, r: float) -> float:
        """

        :param t:
        :param len_t:
        :param supp:
        :param old_size:
        :param old_s:
        :param r:
        :return:
        """
        old_w = len(supp)
        new_size = old_size - 1
        new_s = old_s - len_t
        new_w = old_w
        for item in t:
            if supp[item] == 1:
                new_w -= 1

        return ((old_s * old_size) / (old_w ** r)) - (new_s * new_size) / (new_w ** r)

    def fit_predict(self, X):
        C, clusters, C_sizes, C_sum = self.clope_insert(X, self.r)
        self.labels, self.iters = self.clope_move(X, self.r, C, clusters, C_sizes, C_sum)
        self.labels = np.array(self.labels).astype(int)


if __name__ == '__main__':
    pass