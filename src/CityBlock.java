public class CityBlock extends KNNModel {

    public CityBlock() {
        this(1);
    }

    public CityBlock(int k) {
        super(k);
        metric = Distance.CITYBLOCK;
    }

    @Override
    public double computeDistance(TrainSample x, TrainSample y) {
        double sum = 0.0;
        for (int i = 0; i < x.features.length; i++) {
            sum = sum + Math.abs((x.features[i] - y.features[i]));
        }
        return sum;
    }
}
