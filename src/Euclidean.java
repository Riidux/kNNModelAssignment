public class Euclidean extends KNNModel {

    public Euclidean() {
        this(1);
    }

    public Euclidean(int k) {
        super(k);
        metric = Distance.EUCLIDEAN;
    }

    @Override
    public double computeDistance(TrainSample x, TrainSample y) {
        double sum = 0.0;
        for (int i = 0; i < x.features.length; i++) {
            sum = sum + Math.pow((x.features[i] - y.features[i]), 2);
        }
        return Math.sqrt(sum);
    }
}
