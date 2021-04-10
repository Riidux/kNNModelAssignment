public class Driver {
    public static void main(String[] args) {

        // Test 1
        Euclidean test1 = (Euclidean) KNNFactory.createKNNModel(Distance.EUCLIDEAN, 115);
        test1.addsTrainSampleFromFile("iris.csv");
        test1.addsTrainSampleFromFileWithPredict("iris_test.csv");

        System.out.println("\n ------------------------------------------------------- \n");

        // Test 2
        CityBlock test2 = (CityBlock) KNNFactory.createKNNModel(Distance.CITYBLOCK, 52);
        test2.addsTrainSampleFromFile("iris.csv");
        test2.addsTrainSampleFromFileWithPredict("iris_test.csv");
    }
}
