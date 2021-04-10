import java.util.Arrays;

public class TrainSample {

    double[] features;
    int label;

    public TrainSample(double[] data, int label) {
        this.features = data;
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        return Arrays.toString(features) + ", " + label;
    }
}
