import pandas as pd
import os.path as osp
import numpy as np

def read_product_data():
    return pd.read_csv(osp.join('./data', 'products_train.csv'))

def read_test_data(task):
    return pd.read_csv(osp.join('./data', f'sessions_test_{task}.csv'))

def prepare_submission(predictions, task="task1", check_products=False):
    """
    These tests need to pass as they will also be applied on the evaluator
    """
    test_sessions = read_test_data(task)
    test_locale_names = test_sessions['locale'].unique()
    for locale in test_locale_names:
        sess_test = test_sessions.query(f'locale == "{locale}"')
        preds_locale =  predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
        assert sorted(preds_locale.index.values) == sorted(sess_test.index.values), f"Session ids of {locale} doesn't match"

        if check_products:
            # This check is not done on the evaluator
            # but you can run it to verify there is no mixing of products between locales
            # Since the ground truth next item will always belong to the same locale
            # Warning - This can be slow to run
            products = read_product_data().query(f'locale == "{locale}"')
            predicted_products = np.unique( np.array(list(preds_locale["next_item_prediction"].values)) )
            assert np.all( np.isin(predicted_products, products['id']) ), f"Invalid products in {locale} predictions"
        # Its important that the parquet file you submit is saved with pyarrow backend
    predictions.to_parquet(f'submission_{task}.parquet', engine='pyarrow')

