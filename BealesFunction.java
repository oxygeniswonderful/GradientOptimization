import java.util.ArrayList;

public class BealesFunction implements Function{
    @Override
    public double function(double x, double y) {
        //функция beales
        return Math.pow((1.5 - x + x * y), 2) + Math.pow((2.25 - x + x * Math.pow(y, 2)), 2) + Math.pow((2.625 - x + x * Math.pow(y, 3)), 2);
    }

    @Override
    public ArrayList<Double> funcJacobian(double x, double y) {
        //якобиан функции beales
        ArrayList<Double> jacobian = new ArrayList<>();
        double dz_dx = -12.75 + 3 * y + 4.5 * Math.pow(y, 2) + 5.25 * Math.pow(y, 3) + 2 * x * (3 - 2 * y - Math.pow(y, 2) - 2 * Math.pow(y, 3) + Math.pow(y, 4) + Math.pow(y, 6));
        double dz_dy = 6 * x * (x * (Math.pow(y, 5) + 2.0 / 3.0 * Math.pow(y, 3) - Math.pow(y, 2) - 1.0 / 3.0 * y - 1.0 / 3.0) + 2.625 * Math.pow(y, 2) + 1.5 * y + 0.5);

        jacobian.add(dz_dx);
        jacobian.add(dz_dy);

        return jacobian;
    }

    @Override
    public Double[][] funcHessian(double x, double y) {
        //гессиан функции beales
        Double[][] hessian = new Double[2][2];
        double d2z_dx2 = 2 * (Math.pow(y, 6) + Math.pow(y, 4) - 2*Math.pow(y, 3) - Math.pow(y, 2) - 2*y +3);
        double d2z_dy2 = 6 * x * (x * (5*Math.pow(y, 4) + 2*Math.pow(y, 2)-2*y-1.0/3.0) + 5.25*y + 1.5);
        double d2z_dx_dy = 3 + 9*y + 15.75*Math.pow(y, 2) + 4*x*(-1 - y - 3*Math.pow(y, 2) + 2*Math.pow(y, 3) + 3*Math.pow(y, 5));
        double d2z_dy_dx = 3 + 9*y + 15.75*Math.pow(y, 2) + x*(-4 - 4*y - 12*Math.pow(y, 2) + 8*Math.pow(y, 3) + 12*Math.pow(y, 5));

        hessian[0][0] = d2z_dx2;
        hessian[0][1] = d2z_dx_dy;
        hessian[1][0] = d2z_dy_dx;
        hessian[1][1] = d2z_dy2;

        return hessian;
    }
}
