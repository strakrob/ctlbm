#include "lbm.cuh"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace lbm {

void write_vti(
    const std::string& filename,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Unable to open VTI output file: " + filename);
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"0 " << (cfg.nx - 1) << " 0 " << (cfg.ny - 1) << " 0 " << (cfg.nz - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"0 " << (cfg.nx - 1) << " 0 " << (cfg.ny - 1) << " 0 " << (cfg.nz - 1) << "\">\n";
    out << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"rho\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < rho.size(); ++i) {
        out << std::setprecision(16) << rho[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"ux\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < ux.size(); ++i) {
        out << std::setprecision(16) << ux[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"uy\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < uy.size(); ++i) {
        out << std::setprecision(16) << uy[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"uz\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < uz.size(); ++i) {
        out << std::setprecision(16) << uz[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < ux.size(); ++i) {
        out << std::setprecision(16) << ux[i] << ' ' << uy[i] << ' ' << uz[i] << ' ';
        if ((i + 1) % 2 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "      </PointData>\n";
    out << "      <CellData/>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

}  // namespace lbm
