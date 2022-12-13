import java.util.ArrayList;

public interface Function {

    /**
     * Function of two points
     */
    double function(double x, double y);

    /**
     * Jacobian of function
     */
    ArrayList<Double> funcJacobian(double x, double y);

    /**
     * Hessian of function
     * @return
     */
    Double[][] funcHessian(double x, double y);

}
