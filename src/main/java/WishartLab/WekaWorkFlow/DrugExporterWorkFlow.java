package WishartLab.WekaWorkFlow;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.RandomForest;
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
		
		CostSensitiveClassifier classifier = new CostSensitiveClassifier();
		classifier.setCostMatrix(cm);
		
		// set base learner = random forest
		RandomForest rf = new RandomForest();
		classifier.setClassifier(rf);
		
		// build classifier
//		classifier.buildClassifier(dataset);
			
		// do evaluation on current dataset and classifier
		Evaluation evaluation = new Evaluation(dataset);
		// this evaluation is based on classifier that used under cross validation; 
		evaluation.crossValidateModel(classifier, dataset, 10, new Random(1));
		System.out.println("====================================================");
		System.out.println("Evaluation result:");
	    System.out.println(evaluation.toSummaryString());
	    System.out.println("====================================================");
	    
	    double[][] confusionmatrix = evaluation.confusionMatrix();
	    System.out.println(confusionmatrix[0][0]); // TP
	    System.out.println(confusionmatrix[0][1]); // FP
	    System.out.println(confusionmatrix[1][0]); // FN
	    System.out.println(confusionmatrix[1][1]); // TN
	    
	    double auroc = evaluation.areaUnderROC(dataset.numAttributes()-1);
	    String detailedMatrix = evaluation.toClassDetailsString();
	    
	    System.out.println(String.format("AUROC is => %d", auroc));
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
    		try {
				Instances dataset = dewf.ConvertCSVToInstances("/Users/xuan/Desktop/TestingFile.csv");
				Instances dataset_filtered = dewf.AttributeFilteringEngineering(dataset);
				
				
				// set cost matrix
				CostMatrix costmatrix = new CostMatrix(2);
				costmatrix.setCell(0, 1, 1);
				costmatrix.setCell(1, 0, 1);
				
				Classifier classified = dewf.GenerateCostSensitiveClassifier(dataset_filtered,costmatrix);
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    }
}
