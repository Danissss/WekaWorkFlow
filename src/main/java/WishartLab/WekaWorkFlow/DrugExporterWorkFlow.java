package WishartLab.WekaWorkFlow;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
 * Hello world!
 *
 */
public class DrugExporterWorkFlow 
{
	
	
	
	
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
	
	
	
	public void CalculateAllMatrix(double a, double b, double c, double d) {
		
		
		double sensitivity = a / (a+b);
	    double specificity = d / (c+d);
	    double postive_likelihood_rate = sensitivity /  (1 - specificity);
	    double negative_likelihood_rate = (1 - sensitivity) / specificity;
	    double postive_predictive_value = a / (a + c);
	    double negative_predictive_value = d / (b + d);
	    double accuracy = (a+d)/(a+b+c+d);
	    
	    System.out.println("====================================================");
	    System.out.println(String.format("sensitivity               = %.4f", sensitivity));
	    System.out.println(String.format("specificity               = %.4f", specificity));
	    System.out.println(String.format("postive_likelihood_rate   = %.4f", postive_likelihood_rate));
	    System.out.println(String.format("negative_likelihood_rate  = %.4f", negative_likelihood_rate));
	    System.out.println(String.format("postive_predictive_value  = %.4f", postive_predictive_value));
	    System.out.println(String.format("negative_predictive_value = %.4f", negative_predictive_value));
	    System.out.println(String.format("accuracy                  = %.4f", accuracy));
	    System.out.println("====================================================");
	    
	}
	
	
	public void PerformTestEvaluation(Instances dataset, Instances testset, Classifier clf) throws Exception {
		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
		testset.setClass(testset.attribute(testset.numAttributes()-1));
		
		Evaluation evaluation = new Evaluation(testset);
		
		evaluation.evaluateModel(clf, testset);
		
		
		System.out.println("====================================================");
		System.out.println("Evaluation result on testing set:");
		System.out.println(evaluation.toSummaryString());
		String detailedMatrix = evaluation.toClassDetailsString();
		System.out.println("====================================================");
		AttributeStats attStats = testset.attributeStats(testset.numAttributes()-1);
		System.out.println("Instance statistics:");
		System.out.println(attStats.toString());
		
		
//	    System.out.println(String.format("AUROC is => %d", auroc));
	    System.out.println(String.format("detailed matrix is => %s", detailedMatrix));
		double[][] confusionmatrix = evaluation.confusionMatrix();
		
		double test_a = confusionmatrix[0][0];
		double test_c = confusionmatrix[0][1];
		double test_b = confusionmatrix[1][0];
	    double test_d = confusionmatrix[1][1];
	    System.out.println("True Positive: " + confusionmatrix[0][0] + "| False Positive: " + confusionmatrix[0][1] ); // TP
//	    System.out.println(); // FP
	    System.out.println("False Negative: " +confusionmatrix[1][0] + "| True Negative: " +confusionmatrix[1][1]); // FN
//	    System.out.println(); // TN
	    CalculateAllMatrix(test_a,test_b,test_c,test_d);
	    System.out.println("====================================================");
	}

	/**
	 * running cost sensitive based on given dataset, cost matrix and classifier
	 * @param dataset
	 * @param cm
	 * @param cf
	 * @return
	 * @throws Exception 
	 */
	public Classifier GenerateCostSensitiveClassifier(Instances dataset, CostMatrix cm) throws Exception {
		
		
		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
		System.out.println(dataset.attribute(dataset.numAttributes()-1).toString());
		AttributeStats attStats = dataset.attributeStats(dataset.numAttributes()-1);
		System.out.println("====================================================");
		System.out.println("Instance statistics:");
		System.out.println(attStats.toString());
		
		
		// create new attribute order
//		FastVector class_vec = new FastVector(2);
//		class_vec.addElement("substrate");
//		class_vec.addElement("non-substrate");
//		Attribute reordered_attribute = new Attribute("Class",class_vec);
//		dataset.deleteAttributeAt(dataset.numAttributes() - 1);
//		dataset.insertAttributeAt(reordered_attribute, dataset.numAttributes());
//		dataset.setClass(dataset.attribute(dataset.numAttributes()-1));
//		System.out.println(dataset.classAttribute().toString());
		
		CostSensitiveClassifier classifier = new CostSensitiveClassifier();
		classifier.setCostMatrix(cm);
		
		// set base learner = random forest
		RandomForest rf = new RandomForest();
		classifier.setClassifier(rf);
		
		// build classifier
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
	    System.out.println("True Positive: " + confusionmatrix[0][0] + "| False Positive: " + confusionmatrix[0][1] ); // TP
//	    System.out.println(); // FP
	    System.out.println("False Negative: " +confusionmatrix[1][0] + "| True Negative: " +confusionmatrix[1][1]); // FN
//	    System.out.println(); // TN
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
		return dataset;
	}
	
	
	
	
	
    public static void main( String[] args )
    {
    	DrugExporterWorkFlow dewf = new DrugExporterWorkFlow();
		String current_dir = System.getProperty("user.dir");
		try {
			Instances dataset = dewf.ConvertCSVToInstances(String.format("%s/Dataset/%s", current_dir, 
					"Canalicular_multispecific_organic_anion_transporter_1_MRP2_non_duplicate_substrate_3DFile_3D_descriptor_value_training.csv"));
			Instances dataset_filtered = dewf.AttributeFilteringEngineering(dataset);
			
			
			// set cost matrix
			CostMatrix costmatrix = new CostMatrix(2);
			costmatrix.setCell(0, 1, 1.0);
			costmatrix.setCell(1, 0, 3.0);
			
			Classifier classified = dewf.GenerateCostSensitiveClassifier(dataset_filtered,costmatrix);
			weka.core.SerializationHelper.write(String.format("%s/model/%s.model", current_dir,"BCRPsubstrate"), classified);
			
			// do testing 
			Instances trainingset = dewf.ConvertCSVToInstances(String.format("%s/Dataset/%s", current_dir, 
					"Canalicular_multispecific_organic_anion_transporter_1_MRP2_non_duplicate_substrate_3DFile_3D_descriptor_value_testing.csv"));
			dewf.PerformTestEvaluation(dataset_filtered, trainingset, classified);
			
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
