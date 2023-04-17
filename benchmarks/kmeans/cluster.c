/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <approx.h>
//#include <approx_debug.h>
#include <omp.h>
#include "kmeans.h"
#include <iostream>
#include <fstream>

float	min_rmse_ref = FLT_MAX;		
extern double wtime(void);
/* reference min_rmse value */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,         /* number of data points */
            int      nfeatures,       /* number of attributes for each point */
            float  **features,        /* array: [npoints][nfeatures] */                  
            int      min_nclusters,   /* range of min to max number of clusters */
            int		 max_nclusters,
            float    threshold,       /* loop terminating factor */
            int     *best_nclusters,  /* out: number between min and max with lowest RMSE */
            float ***cluster_centres, /* out: [best_nclusters][nfeatures] */
            float	*min_rmse,          /* out: minimum RMSE */
            int		 isRMSE,            /* calculate RMSE */
            int		 nloops,             /* number of iteration for each number of clusters */
            int converg_max           /* maximum number of iterations before convergence failure */
    )
{    
  int		index =0;	/* number of iteration to reach the best RMSE */
  int		rmse;     /* RMSE for each clustering */
  float delta;

  /* current memberships of points  */
  int *membership = (int*) malloc(npoints * sizeof(int));

  /* new memberships of points computed by the device */
  int *membership_OCL = (int*) malloc(npoints * sizeof(int));

  float *feature_swap = (float*) malloc(npoints * nfeatures * sizeof(float));

  float* feature = features[0];

  #pragma omp target data map(to: feature[0:npoints * nfeatures])     \
                       map(alloc: feature_swap[0:npoints * nfeatures], \
                                  membership_OCL[0:npoints])
{

  /* sweep k from min to max_nclusters to find the best number of clusters */
  for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
  {
    if (nclusters > npoints) break;	/* cannot have more clusters than points */

    int c = 0;  // for each cluster size, count the actual number of loop interations

    #if defined(UNCOALESCED)
    #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE) nowait
    for (int tid = 0; tid < npoints; tid++) {
      for(int i = 0; i <  nfeatures; i++)
        feature_swap[tid * nfeatures + i] = feature[tid * nfeatures + i];
    }
    #else
    // copy the feature to a feature swap region
    #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE) nowait
    for (int tid = 0; tid < npoints; tid++) {
      for(int i = 0; i <  nfeatures; i++)
        feature_swap[i*npoints + tid] = feature[tid * nfeatures + i];
    }
    #endif // ifdef APPROX

    // create clusters of size 'nclusters'
    float** clusters;
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (int i=1; i<nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;

    /* initialize the random clusters */
    int* initial = (int *) malloc (npoints * sizeof(int));
    for (int i = 0; i < npoints; i++) initial[i] = i;
    int initial_points = npoints-1;

    /* iterate nloops times for each number of clusters */
    for(int lp = 0; lp < nloops; lp++)
    {
      int n = 0;

      //if (nclusters > npoints) nclusters = npoints;

      /* pick cluster centers based on the initial array 
         Maybe n = (int)rand() % initial_points; is more straightforward
         without using the initial array
       */	
      for (int i=0; i<nclusters && initial_points >= 0; i++) {

        for (int j=0; j<nfeatures; j++)
          clusters[i][j] = features[initial[n]][j];	// remapped

        /* swap the selected index with the end index. For the next iteration
           of nloops, initial[0] is differetn from initial[0] in the previous iteration */

        int temp = initial[n];
        initial[n] = initial[initial_points];
        initial[initial_points] = temp;
        initial_points--;
        n++;
      }

      /* initialize the membership to -1 for all */
      for (int i=0; i < npoints; i++) membership[i] = -1;

      /* allocate space for and initialize new_centers_len and new_centers */
      int* new_centers_len = (int*) calloc(nclusters, sizeof(int));
      float** new_centers    = (float**) malloc(nclusters *            sizeof(float*));
      new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
      for (int i=1; i<nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;

      char *tpb = std::getenv("THREADS_PER_BLOCK");
      char *nblock = std::getenv("NUM_BLOCKS");
      int NTHREADS = atoi(tpb);
      int NUM_TEAMS = atoi(nblock);

      /* iterate until convergence */

      auto start = omp_get_wtime();
      int loop = 0;
      do {

        delta = 0.0;

        float* cluster = clusters[0];
		int point_id = 0;
        #pragma omp target data map(to: cluster[0:nclusters * nfeatures], index)
		{
        //@APPROX LABEL("entire_perfo") APPROX_TECH(sPerfo|lPerfo)
        #pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) thread_limit(NTHREADS)
        for (point_id = 0; point_id < npoints; point_id++) {
          float min_dist=FLT_MAX;
          //@APPROX LABEL("entire_memo_in") APPROX_TECH(MEMO_IN) IN(feature_swap[point_id:nfeatures:npoints]) OUT(index)
          //@APPROX LABEL("entire_memo_out") APPROX_TECH(MEMO_OUT) IN(feature_swap[0]) OUT(index)
          {
            //@APPROX LABEL("find_nearest_point_perfo") APPROX_TECH(sPerfo|lPerfo)
            for (int i=0; i < nclusters; i++) {
              float dist = 0;
              float ans  = 0;

              #if defined(UNCOALESCED)
              for (int l=0; l< nfeatures; l++) {
                ans += (feature_swap[point_id*nfeatures + l] - cluster[i*nfeatures+l])* 
                  (feature_swap[point_id*nfeatures + l] - cluster[i*nfeatures+l]);
              }
              #else
              for (int l=0; l< nfeatures; l++) {
                ans += (feature_swap[l*npoints+point_id] - cluster[i*nfeatures+l])* 
                  (feature_swap[l*npoints+point_id] - cluster[i*nfeatures+l]);
              }
              #endif //ifdef APPROX
              dist = ans;
              if (dist < min_dist) {
                min_dist = dist;
                index    = i;
              }
            }
          }
          membership_OCL[point_id] = index;
        }
		}
              #pragma omp target update from (membership_OCL[0:npoints])

        /* 
           1 compute the 'new' size and center of each cluster 
           2 update the membership of each point. 
         */

        for (int i = 0; i < npoints; i++)
        {
          int cluster_id = membership_OCL[i];
          new_centers_len[cluster_id]++;
          if (membership_OCL[i] != membership[i])
          {
            delta++;
            membership[i] = membership_OCL[i];
          }
          for (int j = 0; j < nfeatures; j++)
          {
            new_centers[cluster_id][j] += features[i][j];
          }
        }

        /* replace old cluster centers with new_centers */
        for (int i=0; i<nclusters; i++) {
          //printf("length of new cluster %d = %d\n", i, new_centers_len[i]);
          for (int j=0; j<nfeatures; j++) {
            if (new_centers_len[i] > 0)
              clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
            new_centers[i][j] = 0.0;	/* set back to 0 */
          }
          new_centers_len[i] = 0;			/* set back to 0 */
        }	 
        c++;
      } while ((delta > threshold) && (loop++ < converg_max));	/* makes sure loop terminates */
      auto end = omp_get_wtime();
      auto time = end-start;
      std::cout << "Kmeans core timing: " << time << " s" <<  std::endl;
      std::cout << "Kmeans converged in loops: " << loop << std::endl;

      free(new_centers[0]);
      free(new_centers);
      free(new_centers_len);

      /* find the number of clusters with the best RMSE */
      if(isRMSE)
      {
        rmse = rms_err(features,
            nfeatures,
            npoints,
            clusters,
            nclusters);

        if(rmse < min_rmse_ref){
          min_rmse_ref = rmse;			//update reference min RMSE
          *min_rmse = min_rmse_ref;		//update return min RMSE
          *best_nclusters = nclusters;	//update optimum number of clusters
          index = lp;						//update number of iteration to reach best RMSE
        }
      }
    }

    // free the previous cluster centers before using the updated one
    if (*cluster_centres) {
      free((*cluster_centres)[0]);
      free(*cluster_centres);
    }
    *cluster_centres = clusters;

    free(initial);
  }
}
#pragma omp target update from (membership_OCL[0:npoints])
 #ifdef APPROX
 std::string mfile_name = "";
 char *mf_name = std::getenv("MEMBERSHIP_FILENAME");
 if(mf_name)
   mfile_name = mf_name;
 else
   mfile_name = "assignments_approx.txt";
 FILE *membership_file = fopen(mfile_name.c_str(), "w");
 #else
 FILE *membership_file = fopen("assignments_exact.txt", "w");
#endif
 for(int i = 0; i < npoints; i++)
   {
     fprintf(membership_file, "%d\n", membership_OCL[i]);
   }


 fclose(membership_file);
  free(membership_OCL);
  free(feature_swap);
  free(membership);

  #ifdef APPROX_DEV_STATS
  std::ofstream out_file;
  std::string stat_outfile = "";
  char *stat_outfile_ptr = std::getenv("APPROX_STATS_FILE");
  if(stat_outfile_ptr == nullptr)
    stat_outfile = "thread_statistics.csv";
  else
    stat_outfile = stat_outfile_ptr;
  out_file.open(stat_outfile);
  writeDeviceThreadStatistics(out_file);
  out_file.close();
  #endif //APPROX_DEV_STATS


  return index;
}

