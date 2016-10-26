#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>

#define NUMPAT 211
#define NUMIN  13
#define NUMHID 13
#define NUMOUT 3

#define rando() ((double)rand()/(RAND_MAX)+1)
//#define rando1() ((double)(rand()%2))
//#define rando2() ((double)(rand()%2))

void SaveWeights1(double array[][NUMHID+1],int n,int m, char arrName[]){
 
char filename[20];
strcpy (filename,arrName);
 
FILE *fp;
int i,j;
 
 
fp=fopen(filename,"w+");
  
for(i=0;i<n;i++){
 
    for(j=0;j<m;j++)
 
        fprintf(fp,"%f, ",array[i][j]);
        fprintf(fp,"\n");
    }
 
fclose(fp);
 
printf("\n %sfile created",filename);
 
}

void SaveWeights2(double array[][NUMOUT+1],int n,int m, char arrName[]){
 
char filename[20];
strcpy (filename,arrName);
 
FILE *fp;
int i,j;
 
 
fp=fopen(filename,"w+");
  
for(i=0;i<n;i++){
 
    for(j=0;j<m;j++)
 
        fprintf(fp,"%f, ",array[i][j]);
        fprintf(fp,"\n");
 
    }
 
fclose(fp);
 
printf("\n %sfile created",filename);
 
}


int main()
{
    int randPrint;
    int i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    float Input[NUMPAT+1][NUMIN+1];
    float Target[NUMPAT+1][NUMOUT+1];
    
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error, eta = 0.1, alpha = 0.4, smallwt = 0.5;

     /*reading from csv files and storing it in 2D array*/
     char buffer[1024] ;
     char tbuffer[1024] ;
     char *record,*line, *trecord, *tline;
     int x=0,y=0,l=0,m=0;
     FILE *fstream = fopen("cleveland.csv","r");
     FILE *tstream = fopen("test.csv","r");

     if(fstream == NULL || tstream == NULL)
     {
       printf("\n file opening failed ");
       return -1 ;
     }

     while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL && (tline=fgets(tbuffer,sizeof(tbuffer),tstream))!=NULL)
     {
       y=0;
       m=0;
       record = strtok(line,",");

       while(record != NULL)
       {
        //here you can put the record into the array as per your requirement.
        Input[x][y++] = atof(record) ;
        record = strtok(NULL,",");
       }
       trecord = strtok(tline,",");
       while (trecord != NULL)
       { 
        Target[l][m++]= atof(trecord);
        trecord = strtok(NULL,",");       
       }
       ++l ;
       ++x ; 
     }


	/*for (i=0;i<NUMPAT+1;i++)
	{
		for (j=0;j<NUMIN+1;j++)
		{
			printf ("Input[%d][%d] = %f\n",i,j,Input[i][j]);
		}
		printf ("\n");
	}

        printf ("\n");
        printf ("\n");

	for (i=0;i<NUMPAT+1;i++)
	{
		for (j=0;j<NUMOUT+1;j++)
		{
			printf ("Target[%d][%d] = %f\n",i,j,Target[i][j]);
		}
		printf ("\n");
	}*/
    fclose(tstream);
    fclose(fstream);
    
   
    
     for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            //randPrint=rando();
            printf ("randPrint = %f\n",rando());
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    for( epoch = 0 ; epoch < 100000 ; epoch++) {    /* iterate weight updates */
 
        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of individuals */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            
            int k=rando();
            //if (epoch%100 == 0)
            	//printf("rando()=%d\n",k);
            np = p + k * ( NumPattern + 1 - p ) ;

            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */

            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */


                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                
                Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE i.e. stochastic series expansion*/
/*             Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
        }
 

        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f\n", epoch, Error) ;
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
    }
    
    SaveWeights1(WeightIH, NUMIN+1, NUMHID+1, "WeightsIH.csv");
    SaveWeights2(WeightHO, NUMHID+1, NUMOUT+1, "WeightsHO.csv");

    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
/*    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    } */
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "\tTarget%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {
    fprintf(stdout, "\n%d\t", p) ;
      /*  for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }*/
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;

    return 1 ;
}

