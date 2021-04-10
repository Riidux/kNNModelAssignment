import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public abstract class KNNModel {

    protected ArrayList<TrainSample> trainData;
    public Distance metric;
    //protected fields can only be inherited once
    int k; // number of neighbors to account for - k part of nearest neighbor

    public KNNModel(int k) {
        this.trainData = new ArrayList<>();
        setK(k);
    }

    //default
    public KNNModel() {
        this(1);
    }

    public void setK(int k) {
        if (k > 0) {
            this.k = k;
        }
    }

    public void addSingleTrainData(TrainSample oneTrainData) {
        this.trainData.add(oneTrainData);
    }

    public void addMultipleTrainData(ArrayList<TrainSample> multiTrainData) {
        this.trainData.addAll(multiTrainData);
    }

    public abstract double computeDistance(TrainSample x, TrainSample y);


    public static double computeMax(double[] maxArray) {
        double max = maxArray[0];
        for (int i = 0; i < maxArray.length; i++) {
            if (maxArray[i] > max) {
                max = maxArray[i];
            }
        }
        return max;
    }

    public static double computeMode(double[] modeArray) {
        double[] frequency = new double[(int) Math.ceil(computeMax(modeArray)) + 1];
        for (int t = 0; t < (int) Math.ceil(computeMax(modeArray)); t++) {
            frequency[t] = 0;
        }
        for (int r = 0; r < modeArray.length; r++) {
            frequency[(int) modeArray[r]]++;
        }
        double mode = 0;
        double max = 0;
        Random random = new Random();
        for (int i = 0; i < ((int) Math.ceil(computeMax(modeArray) + 1)); i++) {
            if (frequency[i] > max) {
                max = frequency[i];
                mode = i;
            } else if (frequency[i] == max) {
                if (random.nextInt(2) < 1) {
                    max = frequency[i];
                    mode = i;
                }
            }
        }
        return mode;
    }


    public int predict(TrainSample sample) {
        ArrayList<double[]> distances = new ArrayList<>();
        for (TrainSample otherSample : trainData) {
            double distance = this.computeDistance(sample, otherSample);
            double[] arr = {distance, otherSample.getLabel()};
            distances.add(arr);
        }
        if (distances.size() <= 0) {
            return -1; // no train data to compare to means no prediction
        }

        double[][] sortedDistances = new double[distances.size()][2];
        for (int i = 0; i < distances.size(); i++) {
            sortedDistances[i] = distances.get(i);
        }
        if (distances.size() > 2) {
            for (int i = 0; i < sortedDistances.length; i++) {
                double min = sortedDistances[i][0];
                int indexMin = i;
                for (int j = i + 1; j < sortedDistances.length; j++) {
                    if (sortedDistances[j][0] < min) {
                        min = sortedDistances[j][0];
                        indexMin = j;
                    }
                }
                double[] tempHolder = sortedDistances[i];
                sortedDistances[i] = sortedDistances[indexMin];
                sortedDistances[indexMin] = tempHolder;
            }
        }

        // sorted distances, focus on amount of least distances (val of k)
        // return most common label out of k distances (each label is at the end of each distance)
        if (k <= sortedDistances.length) {
            double[] storedLabels = new double[k];
            for (int i = 0; i < k; i++) {
                storedLabels[i] = sortedDistances[i][1];
            }
            return (int) computeMode(storedLabels);
        }
        return -1; // if not enough neighbors, return -1
    }

    public ArrayList<Integer> predict(ArrayList<TrainSample> samples) {
        ArrayList<Integer> storedLabels = new ArrayList<>();
        for (TrainSample sample : samples) {
            storedLabels.add(predict(sample));
        }
        return storedLabels;
    }

    private TrainSample parseTrainSample(String dataLine) { //takes one line of data, parses line into TrainSample
        String[] arr = dataLine.split(","); // splits into String array via commas
        int labelGrab = Integer.parseInt(arr[arr.length - 1]); // label is always at the end despite number of features
        double[] featuresGrab = new double[arr.length - 1];
        for (int i = 0; i < arr.length - 1; i++) {
            featuresGrab[i] = Double.parseDouble(arr[i]);
        }
        return new TrainSample(featuresGrab, labelGrab);
    }


    public void addsTrainSampleFromFile(String fileName) { // adds instantly
        try {
            Scanner scanner = new Scanner(new File(fileName));
            while (scanner.hasNext()) {
                trainData.add(parseTrainSample(scanner.nextLine()));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void addsTrainSampleFromFileWithPredict(String fileName) { // predicts then adds
        try {
            Scanner scanner = new Scanner(new File(fileName));
            while (scanner.hasNext()) {
                TrainSample temp = parseTrainSample(scanner.nextLine());
                System.out.println("Predicted Label For Sample: " + temp + " -> " + predict(temp));
                trainData.add(temp);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
