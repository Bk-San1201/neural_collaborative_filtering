import src.GMF
import src.evaluate
from src.Dataset import Dataset


def loadGMF():
    dataset = Dataset('../Data/ml-1m')
    model = src.GMF.get_model(dataset.num_users, dataset.num_items, 8)
    model.load_weights('../Pretrain/ml-1m_GMF_8_1501651698.h5')
    hits, ndcgs = src.evaluate.evaluate_model(model, dataset.testRatings, dataset.testNegatives, 10, 1)


if __name__ == '__main__':
    loadGMF()

