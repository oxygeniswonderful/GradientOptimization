import java.util.ArrayList;

public class Optimization {

    private final Function function;
    private final OptimizationStep optimizationStep;


    public Optimization(Function function, OptimizationStep optimizationStep) {
        this.function = function;
        this.optimizationStep = optimizationStep;
    }

    public ArrayList<Double> gradientDescent(double x0, double y0, int maxIter, double lr) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();

        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            newPoint = optimizationStep.gradientDescentStep(x, y, grad_x, grad_y, lr);
            x = newPoint.get(0);
            y = newPoint.get(1);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

    public ArrayList<Double> gradientMomentum(double x0, double y0, int maxIter, double lr, double beta) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        double step_x = 0;
        double step_y = 0;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();

        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            newPoint = optimizationStep.gradientMomentumStep(x, y, grad_x, grad_y, lr, beta, step_x, step_y);
            x = newPoint.get(0);
            y = newPoint.get(1);
            step_x = newPoint.get(2);
            step_y = newPoint.get(3);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

    public ArrayList<Double> adaGrad(double x0, double y0, int maxIter, double lr, double eps) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        double sumOfGradSquared_x = 0;
        double sumOfGradSquared_y = 0;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();

        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            newPoint = optimizationStep.adaGradStep(x, y, grad_x, grad_y, lr, sumOfGradSquared_x, sumOfGradSquared_y, eps);
            x = newPoint.get(0);
            y = newPoint.get(1);
            sumOfGradSquared_x = newPoint.get(2);
            sumOfGradSquared_y = newPoint.get(3);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

    public ArrayList<Double> rmsprop(double x0, double y0, int maxIter, double lr, double eps, double beta) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        double sumOfGradSquared_x = 0;
        double sumOfGradSquared_y = 0;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();


        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            newPoint = optimizationStep.rmspropStep(x, y, grad_x, grad_y, lr, sumOfGradSquared_x, sumOfGradSquared_y, eps, beta);
            x = newPoint.get(0);
            y = newPoint.get(1);
            sumOfGradSquared_x = newPoint.get(2);
            sumOfGradSquared_y = newPoint.get(3);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

    public ArrayList<Double> adam(double x0, double y0, int maxIter, double lr, double eps, double beta1, double beta2) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        double sumOfGradSquared_x = 0;
        double sumOfGradSquared_y = 0;
        double sumOfGrad_x = 0;
        double sumOfGrad_y = 0;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();

        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            newPoint = optimizationStep.adamStep(x, y, grad_x, grad_y, lr, sumOfGradSquared_x, sumOfGradSquared_y, eps,
                    beta1, beta2, sumOfGrad_x, sumOfGrad_y, i);
            x = newPoint.get(0);
            y = newPoint.get(1);
            sumOfGrad_x = newPoint.get(2);
            sumOfGrad_y = newPoint.get(3);
            sumOfGradSquared_x = newPoint.get(4);
            sumOfGradSquared_y = newPoint.get(5);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

    public ArrayList<Double> newton(double x0, double y0, int maxIter) {
        double x = x0;
        double y = y0;
        double grad_x;
        double grad_y;
        Double[][] hessian;
        ArrayList<Double> grad;
        ArrayList<Double> newPoint;
        ArrayList<Double> minimum = new ArrayList<>();

        for (int i = 0; i < maxIter; i++) {
            grad = function.funcJacobian(x, y);
            grad_x = grad.get(0);
            grad_y = grad.get(1);

            hessian = function.funcHessian(x, y);

            newPoint = optimizationStep.newtonStep(x, y, grad_x, grad_y, hessian);
            x = newPoint.get(0);
            y = newPoint.get(1);
        }

        minimum.add(x);
        minimum.add(y);
        minimum.add(function.function(x, y));
        return minimum;
    }

}
