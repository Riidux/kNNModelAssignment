public class KNNFactory {

    public static KNNModel createKNNModel(Distance distance, int k) {
        if (distance == Distance.EUCLIDEAN) {
            return new Euclidean(k);
        }
        return new CityBlock(k);
    }
}