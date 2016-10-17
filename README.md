# flexible-unit-length-embedding
Animate the possible unit-length embeddings of graphs into the plane.

This was inspired by Ed Pegg and Greg Egan's work on the McGee graph.
Pegg found a unit-distance embedding into the plane, Bram Cohen asked if it's flexible,
and Egan showed that it is with nice animation.

For a general background on the graph you can see [John Baez's post](http://blogs.ams.org/visualinsight/2015/09/15/mcgee-graph/).
The related discussion on Stack exchange is [here](http://math.stackexchange.com/questions/1484002/is-unit-mcgee-rigid).

This program implements a random crawl in the configuration space of unit-distance
embeddings. A random tangent vector is chosen, and the program integrates it and
follows the geodesic (in a 9-dimensional submanifold of R^48 that contains all
configurations modulo translations and rotations).
I was quite disappointed to see that the graph turns immediately into a huge mess
and never recovers. Simpler graphs such as a pentagon or 3x3 grid give better results.
