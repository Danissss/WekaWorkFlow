package WishartLab.WekaWorkFlow;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.SwapValues;

/**
 * Hello world!
 *
 */
public class DrugExporterWorkFlow 
{
	
	// weka crossValidateModel function
//	public void crossValidateModel(Classifier classifier, Instances data, int numFolds, Random random, Object... forPrinting) throws Exception {
//
//		// Make a copy of the data we can reorder
//		data = new Instances(data);
//		data.randomize(random);
//		if (data.classAttribute().isNominal()) {
//			data.stratify(numFolds);
//		}
//
//		// We assume that the first element is a
//		// weka.classifiers.evaluation.output.prediction.AbstractOutput object
//		AbstractOutput classificationOutput = null;
//		if (forPrinting.length > 0 && forPrinting[0] instanceof AbstractOutput) {
//			// print the header first
//			classificationOutput = (AbstractOutput) forPrinting[0];
//			classificationOutput.setHeader(data);
//			classificationOutput.printHeader();
//		}
//
//		// Do the folds
//		for (int i = 0; i < numFolds; i++) {
//			Instances train = data.trainCV(numFolds, i, random);
//			setPriors(train);
//			Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
//			copiedClassifier.buildClassifier(train);
//			if (classificationOutput == null && forPrinting.length > 0) {
//				((StringBuffer)forPrinting[0]).append("\n=== Classifier model (training fold " + (i + 1) +") ===\n\n" +
//						copiedClassifier);
//			}
//			Instances test = data.testCV(numFolds, i);
//			if (classificationOutput != null){
//				evaluateModel(copiedClassifier, test, forPrinting);
//			} else {
//				evaluateModel(copiedClassifier, test);
//			}
//		}
//		m_NumFolds = numFolds;
//
//		if (classificationOutput != null) {
//			classificationOutput.printFooter();
//		}
//	}
	
	
	/**
	 * running cost sensitive based on given dataset, cost matrix and classifier
	 * @param dataset
	 * @param cm
	 * @param cf
	 * @return
	 * @throws Exception 
	 */
	public Classifier GenerateCostSensitiveClassifier(Instances dataset) throws Exception {
			
		
		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
		
		CostSensitiveClassifier classifier = new CostSensitiveClassifier();
		
		
		// create default cost matrix of a particular size
		CostMatrix costmatrix = new CostMatrix(2);
		classifier.setCostMatrix(costmatrix);
		
		// set base learner = random forest
		RandomForest rf = new RandomForest();
		classifier.setClassifier(rf);
		
		// build classifier
		classifier.buildClassifier(dataset);
			
		// do evaluation on current dataset and classifier
		Evaluation evaluation = new Evaluation(dataset);
		evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));
		System.out.println("====================================================");
		System.out.println("Evaluation result:");
	    System.out.println(evaluation.toSummaryString());
	    System.out.println("====================================================");
	    
		return classifier;
		
	}
	
	
	/**
	 * 
	 * @param a true positive TP
	 * @param b false negative FN
	 * @param c false positive FP
	 * @param d true negative TN
	 */
	public void CalculateAllMatrix(double a, double b, double c, double d) {
		
		
		double sensitivity = a / (a+b); // aka true_positive_rate
	    double specificity = d / (c+d);
	    double postive_likelihood_rate = sensitivity /  (1 - specificity);
	    double negative_likelihood_rate = (1 - sensitivity) / specificity;
	    double positive_predictive_value = a / (a + c);
	    double negative_predictive_value = d / (b + d);
	    double accuracy = (a+d)/(a+b+c+d);
	    double mcc = (a * d - c * b) / Math.sqrt((a+c)*(a+b)*(d+c)*(d+b));
	    double f1  = 2*((positive_predictive_value * sensitivity)/(positive_predictive_value + sensitivity));
	    
	    System.out.println("====================================================");
	    System.out.println(String.format("sensitivity               = %.4f", sensitivity));
	    System.out.println(String.format("specificity               = %.4f", specificity));
	    System.out.println(String.format("postive_likelihood_rate   = %.4f", postive_likelihood_rate));
	    System.out.println(String.format("negative_likelihood_rate  = %.4f", negative_likelihood_rate));
	    System.out.println(String.format("positive_predictive_value = %.4f", positive_predictive_value));
	    System.out.println(String.format("negative_predictive_value = %.4f", negative_predictive_value));
	    System.out.println(String.format("accuracy                  = %.4f", accuracy));
	    System.out.println(String.format("MCC 						= %.4f", mcc));
	    System.out.println(String.format("F1 score 					= %.4f", f1));
	    System.out.println("====================================================");
	    
	}
	
	/**
	 * 
	 * @param dataset
	 * @param testset
	 * @param clf
	 * @param cost
	 * @throws Exception
	 */
	public void PerformTestEvaluation(Instances dataset, Instances testset, Classifier clf, double cost) throws Exception {
		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
		testset.setClass(testset.attribute(testset.numAttributes()-1));
		// reorder the attribute
		testset = ReorderInstances(testset);
		
		System.out.println("====================================================");
		AttributeStats attStats = testset.attributeStats(testset.numAttributes()-1);
		System.out.println("Instance statistics:");
		System.out.println(attStats.toString());
		
		
		Classifier classifier = new RandomForest();
		classifier.buildClassifier(dataset);

		
		int TP = 0;
		int FN = 0;
		int TN = 0;
		int FP = 0;
		
		for(int i = 0; i < testset.size(); i++) {
			String original_class_string = testset.instance(i).stringValue(testset.numAttributes()-1);
			double[] resultList = classifier.distributionForInstance(testset.instance(i));

			if (resultList[0] >= (1/(1+cost))) {
				// predict as true
				if(!original_class_string.contains("non-")) {
					// True Positive
					TP++;
				}else {
					// False Positive
					FP++;
				}
				
			}else {
				if(original_class_string.contains("non-")) {
					// True Negative
					TN++;
					
				}else {
					// False Negative
					FN++;
				}
			}
			
		}
	    
	    System.out.println("True Positive: " + TP + "| False Positive: " + FP ); // TP
	    System.out.println("False Negative: " + FN + "| True Negative: " + TN); // FN
	    CalculateAllMatrix(TP,FN,FP,TN);
	    
	    
	    
	    System.out.println("====================================================");
	}
	
	
	/**
	 * put non/negative class at end
	 * @param dataset
	 * @return
	 * @throws Exception 
	 */
	public Instances ReorderInstances(Instances dataset) throws Exception {
		
		SwapValues sp = new SwapValues();
//		for(int i =1; i <6; i++) {
//			training = dataset.trainCV(5, i);
//			testing =dataset.testCV(5, i)
//					
//		}
		String attstats = dataset.attribute(dataset.numAttributes()-1).toString();
		String[] attstats_split = attstats.replace("@attribute Class {", "").replace("}", "").split(",");
//		System.out.println(dataset.attribute(dataset.numAttributes() - 1).toString());
		if (attstats_split[0].contains("non-")) {
			sp.setAttributeIndex(Integer.toString(dataset.numAttributes()));
			sp.setFirstValueIndex("first");
			sp.setSecondValueIndex("last");
			sp.setInputFormat(dataset);
			dataset = Filter.useFilter(dataset, sp);
		}
		return dataset;
	}

	/**
	 * running cost sensitive based on given dataset, cost matrix and classifier
	 * @param dataset
	 * @param cm
	 * @param cf
	 * @return
	 * @throws Exception 
	 */
	public Classifier GenerateCostSensitiveClassifier(Instances dataset, CostMatrix cm, double cost) throws Exception {
		
		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
		dataset.randomize(new Random());
		
		
		
		System.out.println(dataset.attribute(dataset.numAttributes()-1).toString());
		AttributeStats attStats = dataset.attributeStats(dataset.numAttributes()-1);
		System.out.println("====================================================");
		System.out.println("Instance statistics:");
		System.out.println(attStats.toString());
//		String attstats = dataset.attribute(dataset.numAttributes()-1).toString();
//		String[] attstats_split = attstats.replace("@attribute Class {", "").replace("}", "").split(",");
//		boolean reverse = false;
//		if(attstats_split[0].contains("non-")) {
//			reverse = true;
//		}
		
		//CV
		int TP = 0;
		int FN = 0;
		int TN = 0;
		int FP = 0;
		int folds = 10;
		for (int n = 0; n < folds; n++) {
			Instances train = dataset.trainCV(folds, n);
//			System.out.println("size of train => " + train.size());
			Instances test = dataset.testCV(folds, n);
//			System.out.println("size of test => " + test.size());
			
			// build the classifier
			CostSensitiveClassifier cv_classifier = new CostSensitiveClassifier();
			cv_classifier.setCostMatrix(cm);
			cv_classifier.setMinimizeExpectedCost(true);
			RandomForest cv_rf = new RandomForest();
			cv_classifier.setClassifier(cv_rf);
			cv_classifier.buildClassifier(train);
			
			
			
//			Evaluation evaluation = new Evaluation(train);
//	        evaluation.evaluateModel(cv_classifier, test);
//	        System.out.println(evaluation.toSummaryString());
	        
	        
			// test the testing set
			for(int i = 0; i < test.size(); i++) {
				String original_class_string = test.instance(i).stringValue(test.numAttributes()-1);
				// System.out.println(original_class_string);
				
		        // false positive is: it is wrong, it predicted right
				// false negative is: it is right, but predicted wrong 
//				test.instance(i).setValue(test.numAttributes()-1, "?");
				
				double result = cv_classifier.classifyInstance(test.instance(i));
				// System.out.println(result);
				// System.out.println("----------------------------------------------------------------------------");
				if(original_class_string.contains("non-") || original_class_string.contains("Non-")) {
					// it is negative
					if(result == 1.0) {
						// it is wrong, it predicted right => false positive
						TN++;
					}else if (result == 0.0) {
						// it is wrong, it predicted wrong => true negative
						FP++;
					}
				}else if(!original_class_string.contains("non-") && !original_class_string.contains("Non-")) {
					// it is positive
					if(result == 1.0) {
						// it is right, predicted right
						FN++;
					}else if (result == 0.0) {
						// false negative is: it is right, but predicted wrong 
						TP++;
					}
					
				}
			}
			
		}
		
		System.out.println("======== manual cross validation ===================");
		System.out.println("True Positive: " + TP +  "| False Positive: " + FP ); // TP
	    System.out.println("False Negative: " + FN + "| True Negative: " + TN); // FN
	    CalculateAllMatrix(TP,FN,FP,TN);
	    System.out.println("====================================================");
	    
	    
	    
	    CostSensitiveClassifier classifier = new CostSensitiveClassifier();
		RandomForest rf = new RandomForest();
		classifier.setClassifier(rf);
		classifier.setCostMatrix(cm);
		classifier.setMinimizeExpectedCost(true);
		classifier.buildClassifier(dataset);
	    
		// do evaluation on current dataset and classifier
		Evaluation evaluation = new Evaluation(dataset);
		// this evaluation is based on classifier that used under cross validation; 
		evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));
		System.out.println("====================================================");
		System.out.println("Evaluation result:");
	    System.out.println(evaluation.toSummaryString());
	    System.out.println("====================================================");
	    
	    
	    
	    
	    double[][] confusionmatrix = evaluation.confusionMatrix();
	    double a = confusionmatrix[0][0];
	    double c = confusionmatrix[0][1];
	    double b = confusionmatrix[1][0];
	    double d = confusionmatrix[1][1];
	    System.out.println("True Positive: " + confusionmatrix[0][0] + "| False Positive: " + confusionmatrix[0][1] ); 
	    System.out.println("False Negative: " +confusionmatrix[1][0] + "| True Negative: " +confusionmatrix[1][1]); 
	    
	    CalculateAllMatrix(a,b,c,d);
	    
	    
	    
	    
	    
//	    double auroc = evaluation.areaUnderROC(0);
	    String detailedMatrix = evaluation.toClassDetailsString();
	    
//	    System.out.println(String.format("AUROC is => %d", auroc));
	    System.out.println(String.format("detailed matrix is => %s", detailedMatrix));
	    
		return classifier;
		
	}
	
	/**
	 * convert csv to instances
	 * @param csvfile
	 * @return
	 * @throws IOException
	 */
	public Instances ConvertCSVToInstances(String csvfile) throws IOException {
		
		CSVLoader csvloader = new CSVLoader();
		csvloader.setSource(new File(csvfile));
		System.out.println("====================================================");
		System.out.println(String.format("%s is loaded.", csvfile));
		Instances data = csvloader.getDataSet();
		
		return data;
		
	}
	
	
	/**
	 * filtering the instances
	 * 1. remove the string attributes
	 * 2. remove the useless attributes
	 * @param dataset
	 * @throws Exception
	 */
	public Instances AttributeFilteringEngineering(Instances dataset) throws Exception {
		
		System.out.println(String.format("Number of attribute before filtering => %d", dataset.numAttributes()));
		RemoveType removetype = new RemoveType();
		removetype.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset, removetype);
		
		System.out.println(String.format("Number of attribute after remove string type => %d", dataset.numAttributes()));
		RemoveUseless removeuseless = new RemoveUseless();
		removeuseless.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset, removeuseless);
		
		System.out.println(String.format("Number of attribute after remove useless type => %d", dataset.numAttributes()));
		
		
		System.out.println("Swap values");
		SwapValues sp = new SwapValues();
		String attstats = dataset.attribute(dataset.numAttributes()-1).toString();
		String[] attstats_split = attstats.replace("@attribute Class {", "").replace("}", "").split(",");
//		System.out.println(dataset.attribute(dataset.numAttributes() - 1).toString());
		if (attstats_split[0].contains("non-") || attstats_split[0].contains("Non-")) {
			sp.setAttributeIndex(Integer.toString(dataset.numAttributes()));
			sp.setFirstValueIndex("first");
			sp.setSecondValueIndex("last");
			sp.setInputFormat(dataset);
			dataset = Filter.useFilter(dataset, sp);
		}
		
		
		
		return dataset;
	}
	
	
	
	
	
    public static void main( String[] args )
    {
    	DrugExporterWorkFlow dewf = new DrugExporterWorkFlow();
		String current_dir = System.getProperty("user.dir");
		try {
			Instances dataset = dewf.ConvertCSVToInstances(String.format("%s/Dataset/%s", current_dir, 
					"Multidrug_resistance_associated_protein_1_MRP1_non_duplicate_substrate_3DFile_3D_descriptor_value_training.csv"));
			Instances dataset_filtered = dewf.AttributeFilteringEngineering(dataset);
			
			
			// set cost matrix
			double cost = 1.0;
			CostMatrix costmatrix = new CostMatrix(2);
			costmatrix.setCell(0, 1, cost);
			costmatrix.setCell(1, 0, 1.0);
			
			Classifier classified = dewf.GenerateCostSensitiveClassifier(dataset_filtered,costmatrix,cost);
			
			
//			System.exit(0);
//			
//			System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
//			System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
//			CostMatrix costmatrix2 = new CostMatrix(2);
//			costmatrix2.setCell(0, 1, 20.0);
//			costmatrix2.setCell(1, 0, 1.0);
//			Classifier classified2 = dewf.GenerateCostSensitiveClassifier(dataset_filtered,costmatrix2);
//			
//			System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
//			System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
//			CostMatrix costmatrix3 = new CostMatrix(2);
//			costmatrix3.setCell(0, 1, 10.0);
//			costmatrix3.setCell(1, 0, 1.0);
//			Classifier classified3 = dewf.GenerateCostSensitiveClassifier(dataset_filtered,costmatrix3);
			
			
			
			
			
			weka.core.SerializationHelper.write(String.format("%s/model/%s.model", current_dir,"BCRPsubstrate"), classified);
			
			// do testing 
			Instances trainingset = dewf.ConvertCSVToInstances(String.format("%s/Dataset/%s", current_dir, 
					"ATP_binding_cassette_sub_family_G_member_2_non_duplicate_substrate_3DFile_3D_descriptor_value_testing.csv"));
			dewf.PerformTestEvaluation(dataset_filtered, trainingset, classified, cost);
			
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
