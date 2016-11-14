/*

                          3D Zernike Moments
    Copyright (C) 2003 by Computer Graphics Group, University of Bonn
           http://www.cg.cs.uni-bonn.de/project-pages/3dsearch/

Code by Marcin Novotni:     marcin@cs.uni-bonn.de

for more information, see the paper:

@inproceedings{novotni-2003-3d,
    author = {M. Novotni and R. Klein},
    title = {3{D} {Z}ernike Descriptors for Content Based Shape Retrieval},
    booktitle = {The 8th ACM Symposium on Solid Modeling and Applications},
    pages = {216--225},
    year = {2003},
    month = {June},
    institution = {Universit\"{a}t Bonn},
    conference = {The 8th ACM Symposium on Solid Modeling and Applications, June 16-20, Seattle, WA}
}
 *---------------------------------------------------------------------------*
 *                                                                           *
 *                                License                                    *
 *                                                                           *
 *  This library is free software; you can redistribute it and/or modify it  *
 *  under the terms of the GNU Library General Public License as published   *
 *  by the Free Software Foundation, version 2.                              *
 *                                                                           *
 *  This library is distributed in the hope that it will be useful, but      *
 *  WITHOUT ANY WARRANTY; without even the implied warranty of               *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU        *
 *  Library General Public License for more details.                         *
 *                                                                           *
 *  You should have received a copy of the GNU Library General Public        *
 *  License along with this library; if not, write to the Free Software      *
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                *
 *                                                                           *
\*===========================================================================*/

//#include "ZernikeDescriptor.h"

template<class T, class TIn>
ZernikeDescriptor<T, TIn>::ZernikeDescriptor (T* _voxels, int _dim, int _order) :
    voxels_ (_voxels), dim_ (_dim), order_ (_order)
{
    ComputeNormalization ();
    NormalizeGrid ();

    ComputeMoments ();
    ComputeInvariants ();
}

template<class T, class TIn>
ZernikeDescriptor<T, TIn>::ZernikeDescriptor (const char* _rawName, int _order) : order_ (_order)
{
    voxels_ =  ReadGrid (_rawName, dim_);

    // scale + translation normalization
    ComputeNormalization ();
    NormalizeGrid ();

    ComputeMoments ();
    ComputeInvariants ();
}

template<class T, class TIn>
void ZernikeDescriptor<T, TIn>::ComputeMoments ()
{
    gm_.Init (voxels_, dim_, dim_, dim_, xCOG_, yCOG_, zCOG_, scale_, order_);
    //gm_.SetTransform (xCOG_, yCOG_, zCOG_, scale_);
    //gm_.Compute ();

    // Zernike moments
    zm_.Init (order_, gm_);
    zm_.Compute ();
}

/**
 * Cuts off the function : the object is mapped into the unit ball according to
 * the precomputed center of gravity and scaling factor. All the voxels remaining
 * outside the unit ball are set to zero.
 */
template<class T, class TIn>
void ZernikeDescriptor<T, TIn>::NormalizeGrid ()
{
    T point[3];

    // it is easier to work with squared radius -> no sqrt required
    T radius = (T)1 / scale_;
    T sqrRadius = radius * radius;

    for (int x=0; x<dim_; ++x)
    {
        for (int y=0; y<dim_; ++y)
        {
            for (int z=0; z<dim_; ++z)
            {
                if (voxels_[(z*dim_ + y)*dim_ + x] != (T)0)
                {
                    point[0] = (T)x - xCOG_;
                    point[1] = (T)y - yCOG_;
                    point[2] = (T)z - zCOG_;

                    T sqrLen = point[0]*point[0] + point[1]*point[1] + point[2]*point[2];
                    if (sqrLen > sqrRadius)
                    {
                        voxels_[(z*dim_ + y)*dim_ + x] = 0.0;
                    }
                }
            }
        }
    }
}

/**
 * Center of gravity and a scaling factor is computed according to the geometrical
 * moments and a bounding sphere around the cog.
 */
template<class T, class TIn>
void ZernikeDescriptor<T, TIn>::ComputeNormalization ()
{
    ScaledGeometricalMoments<T, T> gm (voxels_, dim_, 0.0, 0.0, 0.0, 1.0);

    // compute the geometrical transform for no translation and scaling, first
    // to get the 0'th and 1'st order properties of the function
    //gm.Compute ();

    // 0'th order moments -> normalization
    // 1'st order moments -> center of gravity
    zeroMoment_ = gm.GetMoment (0, 0, 0);
    xCOG_ = gm.GetMoment (1, 0, 0) / zeroMoment_;
    yCOG_ = gm.GetMoment (0, 1, 0) / zeroMoment_;
    zCOG_ = gm.GetMoment (0, 0, 1) / zeroMoment_;

    // scaling, so that the function gets mapped into the unit sphere

    //T recScale = ComputeScale_BoundingSphere (voxels_, dim_, xCOG_, yCOG_, zCOG_);
    T recScale = 2.0 * ComputeScale_RadiusVar (voxels_, dim_, xCOG_, yCOG_, zCOG_);
    if (recScale == 0.0)
    {
        std::cerr << "\nNo voxels in grid!\n";
        exit (-1);
    }
    scale_ = (T)1 / recScale;
}

/**
 * Reads the grid from a binary file containing the float grid values. The
 * TIn type tells waht is the precision of the grid. In this implementstion
 * it is assumed that the dimensions of the grid are equal along each axis.
 */
template<class T, class TIn>
T* ZernikeDescriptor<T, TIn>::ReadGrid (const char* _fname, int& _dim_)
{
    std::ifstream infile (_fname, std::ios_base::binary | std::ios_base::in);
    if (!infile)
    {
        std::cerr << "Cannot open " << _fname << " for reading.\n";
        exit (-1);
    }

    vector<T> tempGrid;
    TIn temp;

    // read the grid values
    while (infile.read ((char*)(&temp), sizeof (TIn)))
    {
        tempGrid.push_back ((T)temp);
    }

    int d = tempGrid.size ();
    double f = pow ((double)d, 1.0/3.0);
    _dim_ = floor (f+0.5);

    T* result = new T [d];
    for (int i=0; i<d; ++i)
    {
        result[i] = tempGrid[i];
    }

    return result;
}

template<class T, class TIn>
double ZernikeDescriptor<T, TIn>::ComputeScale_BoundingSphere (T* _voxels, int _dim, T _xCOG, T _yCOG, T _zCOG)
{
    T max = (T)0;

    // the edge length of the voxel grid in voxel units
    int d = _dim;

    for (int x=0; x<d; ++x)
    {
        for (int y=0; y<d; ++y)
        {
            for (int z=0; z<d; ++z)
            {
                if (_voxels[(z + d * y) * d + x] > 0.9)
                {
                    T mx = (T)x - _xCOG;
                    T my = (T)y - _yCOG;
                    T mz = (T)z - _zCOG;
                    T temp = mx*mx + my*my + mz*mz;

                    if (temp > max)
                    {
                        max = temp;
                    }
                }
            }
        }
    }

    return std::sqrt (max);
}

template<class T, class TIn>
double ZernikeDescriptor<T, TIn>::ComputeScale_RadiusVar (T* _voxels, int _dim, T _xCOG, T _yCOG, T _zCOG)
{
    // the edge length of the voxel grid in voxel units
    int d = _dim;

    int nVoxels = 0;

    T sum = 0.0;

    for (int x=0; x<d; ++x)
    {
        for (int y=0; y<d; ++y)
        {
            for (int z=0; z<d; ++z)
            {
                if (_voxels[(z + d * y) * d + x] > 0.9)
                {
                    T mx = (T)x - _xCOG;
                    T my = (T)y - _yCOG;
                    T mz = (T)z - _zCOG;
                    T temp = mx*mx + my*my + mz*mz;

                    sum += temp;

                    nVoxels++;
                }
            }
        }
    }

    T retval = sqrt(sum/nVoxels);

    return retval;
}


template<class T, class TIn>
void ZernikeDescriptor<T, TIn>::Reconstruct (ComplexT3D& _grid, int _minN, int _maxN, int _minL, int _maxL)
{
    // the scaling between the reconstruction and original grid
    T fac = (T)(_grid.size ()) / (T)dim_;

    zm_.Reconstruct (_grid,         // result grid
                     xCOG_*fac,     // center of gravity properly scaled
                     yCOG_*fac,
                     zCOG_*fac,
                     scale_/fac,    // scaling factor
                     _minN, _maxN,  // min and max freq. components to be reconstructed
                     _minL, _maxL);
}


/**
 * Computes the Zernike moment based invariants, i.e. the norms of vectors with
 * components of Z_nl^m with m being the running index.
 */
template<class T, class TIn>
void ZernikeDescriptor<T,TIn>::ComputeInvariants ()
{
    //invariants_.resize (order_ + 1);
    invariants_.clear ();
    for (int n=0; n<order_+1; ++n)
    {
        //invariants_[n].resize (n/2 + 1);

        T sum = (T)0;
        int l0 = n % 2, li = 0;

        for (int l = n % 2; l<=n; ++li, l+=2)
        {
            for (int m=-l; m<=l; ++m)
            {
                ComplexT moment = zm_.GetMoment (n, l, m);
                sum += std::norm (moment);
            }

            invariants_.push_back (sqrt (sum));
            //invariants_[n][li] = std::sqrt (sum);
        }
    }
}


template<class T, class TIn>
void ZernikeDescriptor<T,TIn>::SaveInvariants (const char* _fName)
{
    std::ofstream outfile (_fName, std::ios_base::binary | std::ios_base::out);

    float temp;

    //assert (invariants_.size () == (order_ + 1));

    // write the invariants
    //for (int n=0; n<order_+1; ++n)
    //{
    //    int l0 = n % 2, li = 0;
    //    for (int l=l0; l<=n; l+=2, ++li)
    //    {
    //        temp = (float)invariants_[n][li];
    //        outfile.write ((char*)(&temp), sizeof(float));
    //    }
    //}

    int dim = invariants_.size ();
    outfile.write ((char*)(&dim), sizeof(int));

    for (int i=0; i<dim; ++i)
    {
        temp = invariants_[i];
        outfile.write ((char*)(&temp), sizeof(float));
    }
}

template<class T, class TIn>
vector<T> ZernikeDescriptor<T,TIn>::GetInvariants ()
{
    return invariants_;
}
