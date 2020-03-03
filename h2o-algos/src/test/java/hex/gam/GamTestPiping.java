package hex.gam;

import hex.gam.GAMModel.GAMParameters.BSType;
import hex.glm.GLMModel;
import org.junit.BeforeClass;
import org.junit.Test;
import water.DKV;
import water.Scope;
import water.TestUtil;
import water.fvec.Frame;

/***
 * Here I am going to test the following:
 * - model matrix formation with centering
 */
public class GamTestPiping extends TestUtil {
  @BeforeClass
  public static void setup() {
    stall_till_cloudsize(1);
  }

  /**
   * This test is to make sure that we carried out the expansion of a gam column to basis functions
   * correctly.  I will compare the following:
   * 1. binvD generation;
   * 2. model matrix that contains the basis function value for each role of the gam column
   * 
   * I compared my results with the ones generated from R mgcv library.
   * 
   */
  @Test
  public void testGamTransformNoCenter() {
    try {
      Scope.enter();
      // test for multinomial
      String[] ignoredCols = new String[]{"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
      String[] gamCols = new String[]{"C6"};
      double[][] knots = new double[1][];
      knots[0] = new double[]{-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290};
      GAMModel model = getModel(GLMModel.GLMParameters.Family.multinomial,
              "smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv", "C11",
              gamCols, ignoredCols, new int[]{5}, new BSType[]{BSType.cr}, false, true, knots);  // do not save Z mat
      Scope.track_generic(model);
      double[][] rBinvD = new double[][]{{1.5605080,
              -3.5620961,  2.5465468, -0.6524143,  0.1074557}, {-0.4210098,  2.5559955, -4.3258597,  2.6228736,
              -0.4319995},  {0.1047194, -0.6357626,  2.6244918, -3.7337994,  1.6403508}};
      
      TestUtil.checkDoubleArrays(model._output._binvD[0], rBinvD, 1e-6); // compare binvD generation
      
      Frame transformedData =  ((Frame) DKV.getGet(model._output._gamTransformedTrain));  // compare gam columns
      Scope.track(transformedData);
      Scope.track(transformedData.remove("C11"));
      Frame rTransformedData =  parse_test_file("smalldata/gam_test/multinomial_10_classes_10_cols_10000_Rows_train_C6Gam.csv");
      Scope.track(rTransformedData);
      TestUtil.assertIdenticalUpToRelTolerance(transformedData, rTransformedData, 1e-4);
    } finally {
      Scope.exit();
    }
  }

  public GAMModel getModel(GLMModel.GLMParameters.Family family, String fileName, String responseColumn,
                         String[] gamCols, String[] ignoredCols, int[] numKnots, BSType[] bstypes, boolean saveZmat,
                           boolean savePenalty, double[][] knots) {
    GAMModel gam=null;
    try {
      Scope.enter();
      Frame train = parse_test_file(fileName);
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols-1;
      if (family.equals(GLMModel.GLMParameters.Family.binomial) || (family.equals(GLMModel.GLMParameters.Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._standardize=false;
      params._family = family;
      params._response_column = responseColumn;
      params._train = train._key;
      params._bs = bstypes;
      params._k = numKnots;
      params._ignored_columns = ignoredCols;
      params._gam_X = gamCols;
      params._train = train._key;
      params._family = family;
      params._knots = knots;
      params._link = GLMModel.GLMParameters.Link.family_default;
      params._saveZMatrix = saveZmat;
      params._saveGamCols = true;
      params._savePenaltyMat = savePenalty;
      gam = new GAM(params).trainModel().get();
    } finally {
      Scope.exit();
    }
    return gam;
  }

  @Test
  public void testAdaptFrame2GAMColumns() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/gam_test/gamDataRegressionTwoFuns.csv");
      Scope.track(train);
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr, BSType.cr};
      parms._k = new int[]{6,6};
      parms._response_column = train.name(3);
      parms._ignored_columns = new String[]{train.name(0), train.name(1), train.name(2)}; // row of ids
      parms._gam_X = new String[]{train.name(1), train.name(2)};
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;
      parms._saveZMatrix = true;
      parms._saveGamCols = true;

      GAMModel model = new GAM(parms).trainModel().get();
      Frame transformedData = ((Frame) DKV.getGet(model._output._gamTransformedTrain));
      Scope.track(transformedData);
      Frame predictF = Scope.track(model.score(train)); // predict with train data
      Scope.track(predictF);
      System.out.println("Wow");
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testGAMGaussian() {
    try {
      Scope.enter();
      Frame train = parse_test_file("./smalldata/glm_test/gaussian_20cols_10000Rows.csv");
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", 
              "C15", "C16", "C17", "C18", "C19", "C20"};
      String[] gamCols = new String[]{"C11", "C12"};
      DKV.put(train);
      Scope.track(train);
      
      GAMModel.GAMParameters parms = new GAMModel.GAMParameters();
      parms._bs = new BSType[]{BSType.cr, BSType.cr};
      parms._k = new int[]{6,6};
      parms._response_column = "C21";
      parms._ignored_columns = ignoredCols;
      parms._gam_X = gamCols;
      parms._train = train._key;
      parms._family = GLMModel.GLMParameters.Family.gaussian;
      parms._link = GLMModel.GLMParameters.Link.family_default;
      parms._saveZMatrix = true;
      parms._saveGamCols = true;
      parms._standardize = true;

      GAMModel model = new GAM(parms).trainModel().get();
      Frame transformedData = ((Frame) DKV.getGet(model._output._gamTransformedTrain));
      Scope.track(transformedData);
/*      Frame predictF = Scope.track(model.score(train)); // predict with train data
      Scope.track(predictF);*/
      System.out.println("Wow");
      Scope.track_generic(model);
    } finally {
      Scope.exit();
    }
  }

  @Test
  public void testStandardizedCoeff() {
    String[] ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", 
            "C17", "C18", "C19", "C20"};
    String[] gamCols = new String[]{"C11", "C12"};
    // test for Gaussian
    testCoeffs(GLMModel.GLMParameters.Family.gaussian, "smalldata/glm_test/gaussian_20cols_10000Rows.csv",
            "C21", gamCols, ignoredCols);
    // test for binomial
    testCoeffs(GLMModel.GLMParameters.Family.binomial, "smalldata/glm_test/binomial_20_cols_10KRows.csv", 
            "C21", gamCols, ignoredCols);
    // test for multinomial
    ignoredCols = new String[]{"C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"};
    gamCols = new String[]{"C6", "C7"};
    testCoeffs(GLMModel.GLMParameters.Family.multinomial,
            "smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv", "C11",
            gamCols, ignoredCols);

  }

  public void testCoeffs(GLMModel.GLMParameters.Family family, String fileName, String responseColumn, 
                         String[] gamCols, String[] ignoredCols) {
    try {
      Scope.enter();
      Frame train = parse_test_file(fileName);
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols-1;
      if (family.equals(GLMModel.GLMParameters.Family.binomial) || (family.equals(GLMModel.GLMParameters.Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GAMModel.GAMParameters params = new GAMModel.GAMParameters();
      params._standardize=false;
      params._family = family;
      params._response_column = responseColumn;
      params._train = train._key;
      params._bs = new BSType[]{BSType.cr, BSType.cr};
      params._k = new int[]{6,6};
      params._ignored_columns = ignoredCols;
      params._gam_X = gamCols;
      params._train = train._key;
      params._family = family;
      params._link = GLMModel.GLMParameters.Link.family_default;
      params._saveZMatrix = true;
      params._saveGamCols = true;
      params._standardize = true;
      GAMModel gam = new GAM(params).trainModel().get();
      Scope.track_generic(gam);
      Frame transformedData = ((Frame) DKV.getGet(gam._output._gamTransformedTrain));
      Scope.track(transformedData);
      numCols = transformedData.numCols()-1;
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedData.vec(ind).mean());
      Frame transformedDataCenter = ((Frame) DKV.getGet(gam._output._gamTransformedTrainCenter));
      Scope.track(transformedDataCenter);
      numCols = transformedDataCenter.numCols()-1;
      System.out.println("Print center gamx");
      for (int ind = 0; ind < numCols; ind++)
        System.out.println(transformedDataCenter.vec(ind).mean());
      Frame predictF = gam.score(transformedData); // predict with train data
      Scope.track(predictF);
      Frame predictRaw = gam.score(train); // predict with train data
      Scope.track(predictRaw);
      TestUtil.assertIdenticalUpToRelTolerance(predictF, predictRaw, 1e-6);
    } finally {
      Scope.exit();
    }
  }


}
