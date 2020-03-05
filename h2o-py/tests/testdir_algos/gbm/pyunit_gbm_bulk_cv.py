import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

# 
def test_gbm_bulk_cv():
    response = "survived"
    titanic = h2o.import_file(path=pyunit_utils.locate("smalldata/gbm_test/titanic.csv"))
    titanic[response] = titanic[response].asfactor()
    predictors = ["survived", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin"]
    train, valid = titanic.split_frame(ratios=[.8], seed=1234)
    titanic_gbm = H2OGradientBoostingEstimator(seed=1234, nfolds=2)
    titanic_models = titanic_gbm.bulk_train(segments=["pclass"],
                                            x=predictors,
                                            y=response,
                                            training_frame=train,
                                            validation_frame=valid)
    print(titanic_models.as_frame()) # FIXME: do some actual checking


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_gbm_bulk_cv)
else:
    test_gbm_bulk_cv()
