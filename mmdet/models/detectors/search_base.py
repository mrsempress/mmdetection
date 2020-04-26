from abc import abstractmethod


class SearchBaseDetector:

    @abstractmethod
    def arch_parameters(self):
        pass

    @abstractmethod
    def genotype(self):
        pass

    @abstractmethod
    def plot(self, g, filename):
        pass

    @abstractmethod
    def rebuild(self):
        pass

    @abstractmethod
    def reset_tau(self, tau):
        pass

    @abstractmethod
    def reset_do_search(self, do_search):
        pass
