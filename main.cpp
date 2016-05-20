#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/local_function.hpp>

#include <iostream>
#include <string>
#include <algorithm>
#include <tclap/CmdLine.h>

#include<cstdlib>
#include <fstream>

#include <pomerol.h>
#include <pomerol/LatticePresets.h>
#include "mpi_dispatcher/mpi_dispatcher.hpp"

extern boost::mpi::environment env;
boost::mpi::communicator comm;

using namespace Pomerol;

/* Auxiliary routines - implemented in the bottom. */
void print_section (const std::string& str);
int SiteIndexF(size_t size_x, size_t x, size_t y);

int main(int argc, char* argv[]) {
    boost::mpi::environment env(argc,argv);
    boost::mpi::communicator comm;

    print_section("Hubbard nxn");

    int size_x, size_y, wn;
    RealType t, mu, U, beta, reduce_tol, coeff_tol;
    bool calc_gf, calc_2pgf;
    int wf_max, wb_max;
    double eta, hbw, step; // for evaluation of GF on real axis

    try { // command line parser
        TCLAP::CmdLine cmd("Hubbard nxn diag", ' ', "");
        TCLAP::ValueArg<RealType> U_arg("U","U","Value of U",true,10.0,"RealType",cmd);
        TCLAP::ValueArg<RealType> mu_arg("","mu","Global chemical potential",false,0.0,"RealType",cmd);
        TCLAP::ValueArg<RealType> t_arg("t","t","Value of t",false,1.0,"RealType",cmd);

        TCLAP::ValueArg<RealType> beta_arg("b","beta","Inverse temperature",true,100.,"RealType");
        TCLAP::ValueArg<RealType> T_arg("T","T","Temperature",true,0.01,"RealType");
        cmd.xorAdd(beta_arg,T_arg);

        TCLAP::ValueArg<size_t> x_arg("x","x","Size over x",false,2,"int",cmd);
        TCLAP::ValueArg<size_t> y_arg("y","y","Size over y",false,2,"int",cmd);

        TCLAP::ValueArg<size_t> wn_arg("","wf","Number of positive fermionic Matsubara Freqs",false,64,"int",cmd);
        TCLAP::ValueArg<size_t> wb_arg("","wb","Number of positive bosonic Matsubara Freqs",false,1,"int",cmd);
        TCLAP::SwitchArg gf_arg("","calcgf","Calculate Green's functions",cmd, false);
        TCLAP::SwitchArg twopgf_arg("","calc2pgf","Calculate 2-particle Green's functions",cmd, false);
        TCLAP::ValueArg<RealType> reduce_tol_arg("","reducetol","Energy resonance resolution in 2pgf",false,1e-5,"RealType",cmd);
        TCLAP::ValueArg<RealType> coeff_tol_arg("","coefftol","Total weight tolerance",false,1e-12,"RealType",cmd);

        TCLAP::ValueArg<RealType> eta_arg("","eta","Offset from the real axis for Green's function calculation",false,0.05,"RealType",cmd);
        TCLAP::ValueArg<RealType> hbw_arg("D","hbw","Half-bandwidth. Default = U",false,0.0,"RealType",cmd);
        TCLAP::ValueArg<RealType> step_arg("","step","Step on a real axis. Default : 0.01",false,0.01,"RealType",cmd);

        cmd.parse( argc, argv );
        U = U_arg.getValue();
        mu = (mu_arg.isSet()?mu_arg.getValue():U/2);
        boost::tie(t, beta, calc_gf, calc_2pgf, reduce_tol, coeff_tol) = boost::make_tuple( t_arg.getValue(), beta_arg.getValue(),
                                                                                            gf_arg.getValue(), twopgf_arg.getValue(), reduce_tol_arg.getValue(), coeff_tol_arg.getValue());
        boost::tie(size_x, size_y) = boost::make_tuple(x_arg.getValue(), y_arg.getValue());
        boost::tie(wf_max, wb_max) = boost::make_tuple(wn_arg.getValue(), wb_arg.getValue());
        boost::tie(eta, hbw, step) = boost::make_tuple(eta_arg.getValue(), (hbw_arg.isSet()?hbw_arg.getValue():2.*U), step_arg.getValue());
        calc_gf = calc_gf || calc_2pgf;
        calc_gf = calc_gf || calc_2pgf;
    }
    catch (TCLAP::ArgException &e)  // catch parsing exceptions
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; exit(1);}

    int L = size_x*size_y;
    INFO("Diagonalization of " << L << "=" << size_x << "*" << size_y << " sites");
    Lattice Lat;

    /* Add sites */
    std::vector<std::string> names(L);
    for (size_t y=0; y<size_y; y++)
        for (size_t x=0; x<size_x; x++)
        {
            size_t i = SiteIndexF(size_x, x, y);
            std::stringstream s; s << i;
            names[i]="S"+s.str();
            Lat.addSite(new Lattice::Site(names[i],1,2));
        };

    INFO("Sites");
    Lat.printSites();

    /* Add interaction on each site*/
    MelemType U_complex;
    U_complex = std::complex<double> (U,0);
    MelemType mu_complex;
    mu_complex = std::complex<double> (mu,0);
    for (size_t i=0; i<L; i++) LatticePresets::addCoulombS(&Lat, names[i], U_complex, -mu_complex);

    /* Add hopping */
    for (size_t y=0; y<size_y; y++) {
        for (size_t x=0; x<size_x; x++) {
            size_t pos = SiteIndexF(size_x, x,y);
            size_t pos_right = SiteIndexF(size_x, (x+1)%size_x,y); /*if (x == size_x - 1) pos_right = SiteIndexF(0,y); */
            size_t pos_up = SiteIndexF(size_x, x,(y+1)%size_y);

            if (size_x > 1) LatticePresets::addHopping(&Lat, std::min(names[pos], names[pos_right]), std::max(names[pos], names[pos_right]), -t);
            if (size_y > 1) LatticePresets::addHopping(&Lat, std::min(names[pos], names[pos_up]), std::max(names[pos], names[pos_up]), -t);
        };
    };

    return 0;
}

int SiteIndexF(size_t size_x, size_t x, size_t y){
    return y*size_x+x;
}
void print_section (const std::string& str)
{
    if (!comm.rank()) {
        std::cout << std::string(str.size(),'=') << std::endl;
        std::cout << str << std::endl;
        std::cout << std::string(str.size(),'=') << std::endl;
    };
}