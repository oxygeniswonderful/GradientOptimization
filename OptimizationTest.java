import org.junit.jupiter.api.Assertions;
import java.util.ArrayList;

class OptimizationTest {

    double error = 2e-2;
    double x0 = -3, y0 = -3;
    double x_min = 0, y_min = 0, f_min = 0;
    double x_min_true = 3.0, y_min_true = 0.5, f_min_true = 0.0;
    long m;
    ArrayList<Double> minimum;
    OptimizationStep optimizationStep = new OptimizationStep();
    Function function = new BealesFunction();
    Optimization optimization = new Optimization(function, optimizationStep);

    @org.junit.jupiter.api.Test
    void gradientDescent() {
        System.out.println("gradientDescent");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.gradientDescent(x0, y0,1000000, 0.00005);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }

    @org.junit.jupiter.api.Test
    void gradientMomentum() {
        System.out.println("gradientMomentum");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.gradientMomentum(x0, y0,1000000, 0.00005, 0.3);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }

    @org.junit.jupiter.api.Test
    void adaGrad() {
        System.out.println("AdaGrad");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.adaGrad(x0, y0,10000100, 0.1, 1e-8);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }

    @org.junit.jupiter.api.Test
    void rmsprop() {
        System.out.println("rmspop");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.rmsprop(x0,y0,1000000, 0.01, 1e-10, 0.9);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }

    @org.junit.jupiter.api.Test
    void adam() {
        System.out.println("Adam");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.adam(x0, y0,1000000, 0.01, 1e-10, 0.9, 0.999);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }

    @org.junit.jupiter.api.Test
    void newton() {
        System.out.println("newton");
        System.out.print("Execution time: ");
        m = System.currentTimeMillis();
        minimum = optimization.newton(x0, y0,1000000);
        System.out.println((double) (System.currentTimeMillis() - m));
        x_min = minimum.get(0);
        y_min = minimum.get(1);
        f_min = minimum.get(2);
        System.out.printf("x_min=%f; y_min=%f; f_min=%f \n", x_min, y_min, f_min);

        Assertions.assertTrue(Math.abs(x_min - x_min_true) <= error);
        Assertions.assertTrue(Math.abs(y_min - y_min_true) <= error);
        Assertions.assertTrue(Math.abs(f_min - f_min_true) <= error);
    }
}