#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/local_function.hpp>

#include <iostream>
#include <string>
#include <sstream>
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
template <typename T1> void savetxt(std::string fname, T1 in);
double FMatsubara(int n, double beta){return M_PI/beta*(2.*n+1);}

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

    int rank = comm.rank();
    if (!rank) {
        INFO("Terms with 2 operators");
        Lat.printTerms(2);

        INFO("Terms with 4 operators");
        Lat.printTerms(4);
    };

    IndexClassification IndexInfo(Lat.getSiteMap());
    IndexInfo.prepare(false); // Create index space
    if (!rank) { print_section("Indices"); IndexInfo.printIndices(); };
    int index_size = IndexInfo.getIndexSize();

    print_section("Matrix element storage");
    IndexHamiltonian Storage(&Lat,IndexInfo);
    Storage.prepare(); // Write down the Hamiltonian as a symbolic formula
    print_section("Terms");
    if (!rank) INFO(Storage);

    Symmetrizer Symm(IndexInfo, Storage);
    Symm.compute(); // Find symmetries of the problem

    StatesClassification S(IndexInfo,Symm); // Introduce Fock space and classify states to blocks
    S.compute();

    Hamiltonian H(IndexInfo, Storage, S); // Hamiltonian in the basis of Fock Space
    H.prepare(); // enter the Hamiltonian matrices
    H.compute(); // compute eigenvalues and eigenvectors

    RealVectorType evals (H.getEigenValues());
    std::sort(evals.data(), evals.data() + H.getEigenValues().size());
    savetxt("spectrum.dat", evals); // dump eigenvalues

    DensityMatrix rho(S,H,beta); // create Density Matrix
    rho.prepare();
    rho.compute(); // evaluate thermal weights with respect to ground energy, i.e exp(-beta(e-e_0))/Z

    INFO("<N> = " << rho.getAverageOccupancy()); // get average total particle number
    savetxt("N_T.dat",rho.getAverageOccupancy());

    // Green's function calculation starts here

    FieldOperatorContainer Operators(IndexInfo, S, H); // Create a container for c and c^+ in the eigenstate basis

    if (calc_gf) {

        INFO("1-particle Green's functions calc");
        std::set<ParticleIndex> f;
        std::set<IndexCombination2> indices2;
        ParticleIndex d0 = IndexInfo.getIndex("S0", 0, down);
        ParticleIndex u0 = IndexInfo.getIndex("S0", 0, up);
        f.insert(u0);
        f.insert(d0);
        for (size_t x = 0; x < size_x; x++) {
            ParticleIndex ind = IndexInfo.getIndex(names[SiteIndexF(size_x, x, 0)], 0, down);
            f.insert(ind);
            indices2.insert(IndexCombination2(d0, ind));
        };

        Operators.prepareAll(f);
        Operators.computeAll(); // evaluate c, c^+ for chosen indices

        GFContainer G(IndexInfo, S, H, rho, Operators);

        G.prepareAll(indices2); // identify all non-vanishing block connections in the Green's function
        G.computeAll(); // Evaluate all GF terms, i.e. resonances and weights of expressions in Lehmans representation of the Green's function

        if (!comm.rank()) { // dump gf into a file
            std::set<IndexCombination2>::iterator ind2;
            for (ind2 = indices2.begin(); ind2 !=
                                          indices2.end(); ++ind2) { // loops over all components (pairs of indices) of the Green's function
                // Save Matsubara GF from pi/beta to pi/beta*(4*wf_max + 1)
                std::cout << "Saving imfreq G" << *ind2 << " on " << 4 * wf_max << " Matsubara freqs. " << std::endl;

                std::stringstream fname;
                fname << "gw_imag" << (*ind2).Index1 << (*ind2).Index2 << ".dat";
                std::ofstream gw_im(fname.str().c_str());

                const GreensFunction &GF = G(*ind2);
                for (int wn = 0; wn < wf_max * 4; wn++) {
                    ComplexType val = GF(
                            I * FMatsubara(wn, beta)); // this comes from Pomerol - see GreensFunction::operator()
                    gw_im << std::scientific << std::setprecision(12) << FMatsubara(wn, beta) << "   " << real(val) <<
                    " " << imag(val) << std::endl;
                };
                gw_im.close();
            }
        }
    }

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

template <typename T1> void savetxt(std::string fname, T1 in){std::ofstream out(fname.c_str()); out << in << std::endl; out.close();};