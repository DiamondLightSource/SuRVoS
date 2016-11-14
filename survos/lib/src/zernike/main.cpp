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

/*
    This is a demonstration of how the classes may be used to generate the
    3D Zernike descriptors from a given input binary file containing the
    voxel grid representation of the object.

    Notice that in the below case, the file contains a cubic grid, i.e. the
    x-, y-, and z-dimensions are equal (such setting should satisfy most needs).
*/

// ---- local includes ----
#include "ZernikeDescriptor.h"

// ---- std includes ----
#include <vector>
#include <complex>
#include <fstream>
#include <sstream>
#include <iostream>

#include <cstring>
#include <stdio.h>

// reads a voxel grid from a binary file
template<class TIn, class TOut>
TOut* ReadGrid (const char* _fname, int& _dim)
{
    std::ifstream infile (_fname, std::ios_base::binary | std::ios_base::in);

    vector<double> tempGrid;
    TIn temp;

    // read the grid values
    while (infile.read ((char*)(&temp), sizeof (TIn)))
    {
        tempGrid.push_back ((double)temp);
    }

    int d = tempGrid.size ();
    double f = std::pow<double> ((double)d, 1.0/3.0);

    _dim = floor (f+0.5);

    TOut* result = new TOut [d];
    for (int i=0; i<d; ++i)
    {
        result[i] = tempGrid[i];
    }

    return result;
}

int main (int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ZernikeMoments " <<
                     "filename " <<
                     "<MaxOrder> " << std::endl;
        return 0;
    }

    //int d;
    //double* voxels = ReadGrid<float, double> (argv[1], d);


    // .inv file name
    char  buf[200];
    char* ch_ptr = strrchr (argv[1], '.');
    char* invFName;
    if (ch_ptr)
    {
        strncpy (buf, argv[1], ch_ptr - argv[1]);
        buf[ch_ptr - argv[1]] = '\0';
    }
    else
    {
        fprintf (stderr, "No extension in input filename? : %s\n", argv[1]);
    }

    invFName = buf;

    // compute the zernike descriptors
    ZernikeDescriptor<double, float> zd (argv[1], atoi (argv[2]));

    strcat (invFName, ".inv");
    std::cout << "Saving invariants file: " << invFName << " \n";

    // save them into an .inv file
    zd.SaveInvariants (invFName);
}
