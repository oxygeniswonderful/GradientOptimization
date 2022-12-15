import java.util.ArrayList;

public class OptimizationStep {

    public ArrayList<Double> gradientDescentStep(double current_x, double current_y, double grad_x, double grad_y, double lr) {
        ArrayList<Double> newPoint = new ArrayList<>();
        current_x -= lr*grad_x;
        current_y -= lr*grad_y;
        newPoint.add(current_x);
        newPoint.add(current_y);
        return newPoint;
    }

    public ArrayList<Double> gradientMomentumStep(double current_x, double current_y, double grad_x, double grad_y, double lr, double beta,
                                              double step_x, double step_y) {
        ArrayList<Double> newPoint = new ArrayList<>();
        step_x = step_x*beta + lr*grad_x;
        step_y = step_y*beta + lr*grad_y;
        newPoint.add(current_x-step_x);
        newPoint.add(current_y-step_y);
        newPoint.add(step_x);
        newPoint.add(step_y);
        return newPoint;
    }

    public ArrayList<Double> adaGradStep(double current_x, double current_y, double grad_x, double grad_y, double lr,
                                     double sumOfGradSquared_x, double sumOfGradSquared_y, double eps) {
        ArrayList<Double> newPoint = new ArrayList<>();
        double step_x, step_y;
        sumOfGradSquared_x += grad_x*grad_x;
        sumOfGradSquared_y += grad_y*grad_y;
        step_x = lr*grad_x / (Math.sqrt(sumOfGradSquared_x) + eps);
        step_y = lr*grad_y / (Math.sqrt(sumOfGradSquared_y) + eps);
        newPoint.add(current_x-step_x);
        newPoint.add(current_y-step_y);
        newPoint.add(sumOfGradSquared_x);
        newPoint.add(sumOfGradSquared_y);
        return newPoint;
    }

    public ArrayList<Double> rmspropStep(double current_x, double current_y, double grad_x, double grad_y, double lr,
                                     double sumOfGradSquared_x, double sumOfGradSquared_y, double eps, double beta) {
        ArrayList<Double> newPoint = new ArrayList<>();
        double step_x, step_y;
        sumOfGradSquared_x = beta*sumOfGradSquared_x + (1-beta)*grad_x*grad_x;
        sumOfGradSquared_y = beta*sumOfGradSquared_y + (1-beta)*grad_y*grad_y;
        step_x = lr*grad_x / (Math.sqrt(sumOfGradSquared_x) + eps);
        step_y = lr*grad_y / (Math.sqrt(sumOfGradSquared_y) + eps);
        current_x -= step_x;
        current_y -= step_y;
        newPoint.add(current_x);
        newPoint.add(current_y);
        newPoint.add(sumOfGradSquared_x);
        newPoint.add(sumOfGradSquared_y);
        return newPoint;
    }

    public ArrayList<Double> adamStep(double current_x, double current_y, double grad_x, double grad_y, double lr,
                                  double sumOfGradSquared_x, double sumOfGradSquared_y, double eps, double beta1,
                                  double beta2, double sumOfGrad_x, double sumOfGrad_y, int currentIter) {
        ArrayList<Double> newPoint = new ArrayList<>();
        double step_x, step_y;
        double sumOfGradCorrected_x, sumOfGradCorrected_y, sumOfGradSquaredCorrected_x, sumOfGradSquaredCorrected_y;

        //compute the first momentum of gradient
        sumOfGrad_x = beta1*sumOfGrad_x + (1 - beta1)*grad_x;
        sumOfGradCorrected_x = sumOfGrad_x / (1 - Math.pow(beta1,
                currentIter+1));

        //reverse weighting of the first moment in order to avoid the tendency of sum_of_grad to 0
        sumOfGrad_y = beta1*sumOfGrad_y + (1 - beta1)*grad_y;
        sumOfGradCorrected_y = sumOfGrad_y / (1 - Math.pow(beta1, currentIter+1));

        //compute the second momentum of gradient
        sumOfGradSquared_x = beta2*sumOfGradSquared_x + (1-beta2)*(grad_x*grad_x);
        sumOfGradSquaredCorrected_x = sumOfGradSquared_x / (1 - Math.pow(beta2, currentIter+1));

        //reverse weighting of the second moment in order to avoid the tendency of sum_of_grad to 0
        sumOfGradSquared_y = beta2*sumOfGradSquared_y + (1-beta2)*(grad_y*grad_y);
        sumOfGradSquaredCorrected_y = sumOfGradSquared_y / (1 - Math.pow(beta2, currentIter+1));

        step_x = lr*sumOfGradCorrected_x / (Math.sqrt(sumOfGradSquaredCorrected_x) + eps);
        step_y = lr*sumOfGradCorrected_y / (Math.sqrt(sumOfGradSquaredCorrected_y) + eps);

        newPoint.add(current_x-step_x);
        newPoint.add(current_y-step_y);
        newPoint.add(sumOfGrad_x);
        newPoint.add(sumOfGrad_y);
        newPoint.add(sumOfGradSquared_x);
        newPoint.add(sumOfGradSquared_y);
        return newPoint;
    }

    public ArrayList<Double> newtonStep(double current_x, double current_y, double grad_x, double grad_y, Double[][] hessian) {
        ArrayList<Double> newPoint = new ArrayList<>();
        Double[][] step;
        Matrix matrix = new Matrix2D();
        Double[][] grad = new Double[2][2];
        double step_x, step_y;

        grad[0][0] = grad_x;
        grad[0][1] = grad_y;
        grad[1][0] = 0.0;
        grad[1][1] = 0.0;

        step = matrix.matrixMultiplication(grad, matrix.reverseMatrix(hessian));
        step_x = step[0][0];
        step_y = step[0][1];

        newPoint.add(current_x-step_x);
        newPoint.add(current_y-step_y);
        return newPoint;
    }
}
