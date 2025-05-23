#include "api.h" 
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Solve problem from a given workspace and measure setup and solve time 
void daqp_solve(DAQPResult *res, DAQPWorkspace *work){
#ifdef PROFILING
    DAQPtimer timer;
    tic(&timer);
#endif
    // Select algorithm
    if(work->settings->eps_prox==0){
        if(work->bnb != NULL)
            res->exitflag = daqp_bnb(work);
        else if(work->nh > 1)
            res->exitflag = daqp_hiqp(work);
        else
            res->exitflag = daqp_ldp(work);

        if(res->exitflag > 0) ldp2qp_solution(work); // Retrieve qp solution 
    }
    else{//Prox
        res->exitflag = daqp_prox(work);
    }
#ifdef PROFILING
    toc(&timer);
#endif

    // Package result
    daqp_extract_result(res,work);
    // Add time to result
#ifdef PROFILING
    res->solve_time = get_time(&timer);
#else
    res->solve_time = 0; 
#endif
}

// Setup and solve problem
void daqp_quadprog(DAQPResult *res, DAQPProblem* qp, DAQPSettings *settings){
    int setup_flag;

    DAQPWorkspace work;
    work.settings = settings;
    setup_flag = setup_daqp(qp,&work,&(res->setup_time));
    res->exitflag = setup_flag;

    if(setup_flag >= 0){
        daqp_solve(res,&work);
        // Free memory
        if(settings != NULL) work.settings = NULL;
        free_daqp_workspace(&work);
        free_daqp_ldp(&work);
    }
}

// Setup workspace and transform QP to LDP
int setup_daqp(DAQPProblem* qp, DAQPWorkspace *work, c_float* setup_time){
    int errorflag;
    int own_settings=1;
    (void)setup_time; // avoids warning when compiling without profiling
#ifdef PROFILING
    DAQPtimer timer;
    if(setup_time != NULL){
        *setup_time = 0; // in case setup fails 
        tic(&timer);
    }
#endif
    // Check if QP is well-posed
    //validate_QP(qp);

    // Count number of soft/binary constraints 
    // (to account for it in allocation)
    int ns = 0, nb = 0;
    int i;
    if(qp->sense != NULL){
        for(i = 0; i < qp->m ; i++){
            if(qp->sense[i] & SOFT) ns++;
            if(qp->sense[i] & BINARY) nb++;
        }
    }
    // Correct number of soft constraints if several hierarchies
    if(qp->nh > 1){
        ns = 0; // Reset
        int start = 0;
        for(int i = 0; i < qp->nh; i++){
            ns = (ns  > qp->break_points[i]-start) ? ns : qp->break_points[i]-start;
            start = qp->break_points[i];
        }
     }
   
    // Setup workspace
    if(work->settings == NULL)
        allocate_daqp_settings(work);
    else
        own_settings = 0;
    allocate_daqp_workspace(work,qp->n,ns);

    errorflag = setup_daqp_ldp(work,qp);
    if(errorflag < 0){
        if(own_settings==0) work->settings = NULL;
        free_daqp_workspace(work);
        return errorflag;
    }

    errorflag = setup_daqp_bnb(work, nb, ns);  
    if(errorflag < 0){
        if(own_settings==0) work->settings = NULL;
        free_daqp_workspace(work);
        return errorflag;
    }

#ifdef PROFILING
    if(setup_time != NULL){
        toc(&timer);
        *setup_time = get_time(&timer);
    }
#endif
    return 1;
}

//  Setup LDP from QP  
int setup_daqp_ldp(DAQPWorkspace *work, DAQPProblem *qp){
    int update_mask = UPDATE_M+UPDATE_d+UPDATE_sense; // Always update M, d and sense
    int error_flag;
    int alloc_R=0, alloc_v=0;

    // Only allocate Rinv if H is not NULL
    if(qp->H!=NULL){
        alloc_R = 1;
        update_mask+=UPDATE_Rinv;
    }
    // Only allocate v if f is not NULL, or if proximal
    if(qp->f!=NULL || work->settings->eps_prox != 0){
        alloc_v = 1;
        update_mask+=UPDATE_v;
    }

    // Allocate memory for LDP
    allocate_daqp_ldp(work, qp->n, qp->m, qp->ms, alloc_R, alloc_v);


    // Update hierarchy if hqp
    if(qp->nh > 1) update_mask += UPDATE_hierarchy;

    // Form LDP
    error_flag = update_ldp(update_mask, work, qp);
    if(error_flag<0){
        free_daqp_ldp(work);
        return error_flag;
    }
    return 1;
}


void setup_daqp_hiqp(DAQPWorkspace* work, int* break_points, int nh){
    if(nh > 1){
        work->nh = nh;
        work->break_points= break_points;
    }
}

int setup_daqp_bnb(DAQPWorkspace* work, int nb, int ns){
    int i, nadded;
    if(nb > work->n) return EXIT_OVERDETERMINED_INITIAL;
    if((work->bnb == NULL) && (nb >0)){
        work->bnb= malloc(sizeof(DAQPBnB));

        work->bnb->nb = nb;
        // Detect which constraints are binary
        work->bnb->bin_ids = malloc(nb*sizeof(int));
        for(i = 0, nadded = 0; nadded < nb; i++){
            if(work->qp->sense[i] & BINARY)
                work->bnb->bin_ids[nadded++] = i;
        }

        // Setup tree
        work->bnb->tree= malloc((work->bnb->nb+1)*sizeof(DAQPNode));
        work->bnb->tree_WS= malloc((work->n+ns+1)*(nb+1)*sizeof(int));
        work->bnb->n_nodes = 0; 
        work->bnb->nWS= 0; 
        work->bnb->fixed_ids= malloc((nb+1)*sizeof(int));
    }
    return 1;
}

// Free data for LDP 
void free_daqp_ldp(DAQPWorkspace *work){
    if(work->sense==NULL) return; // Already freed
    free(work->sense);
    if(work->Rinv != NULL){
        free(work->Rinv);
    }
    if(work->RinvD != NULL){
        free(work->RinvD);
    }
    if(work->v != NULL){
        free(work->v);
    }
    if(work->scaling != NULL){
        free(work->scaling);
        free(work->M);
        free(work->dupper);
        free(work->dlower);
    }

#ifdef SOFT_WEIGHTS
    if(work->d_ls != NULL){
        free(work->d_ls);
        free(work->d_us);
        free(work->rho_ls);
        free(work->rho_us);
    }
#endif

    work->sense = NULL;
}

void allocate_daqp_settings(DAQPWorkspace *work){
    if(work->settings == NULL){
        work->settings = malloc(sizeof(DAQPSettings));
        daqp_default_settings(work->settings);
    }
}

void free_daqp_bnb(DAQPWorkspace* work){
    if(work->bnb != NULL){
        free(work->bnb->tree);
        free(work->bnb->tree_WS);
        free(work->bnb->fixed_ids);
        free(work->bnb);
        work->bnb = NULL;
    }
}

// Allocate memory for iterates  
void allocate_daqp_workspace(DAQPWorkspace *work, int n, int ns){
    work->n = n;
    n = n + ns; //To account for soft_constraints
    work->Rinv = NULL;
    work->RinvD = NULL;
    work->v = NULL;
    work->scaling = NULL;

    work->lam = malloc((n+1)*sizeof(c_float));
    work->lam_star = malloc((n+1)*sizeof(c_float));

    work->WS= malloc((n+1)*sizeof(int));

    work->D= malloc((n+1)*sizeof(c_float));
    work->xldl= malloc((n+1)*sizeof(c_float));
    work->zldl= malloc((n+1)*sizeof(c_float));
    work->L= malloc(((n+1)*(n+2)/2)*sizeof(c_float));



    work->u= malloc(work->n*sizeof(c_float));
    work->x = work->u; 

    work->xold= malloc(work->n*sizeof(c_float));

#ifdef SOFT_WEIGHTS
    work->d_ls= NULL;
    work->d_us= NULL;
    work->rho_ls= NULL;
    work->rho_us= NULL;
#endif

    work->bnb = NULL;
    work->nh = 0;
    work->break_points = NULL;
    reset_daqp_workspace(work);
}

void allocate_daqp_ldp(DAQPWorkspace *work, int n, int m, int ms, int alloc_R, int alloc_v){
    int i;
    // Always allocate scaling, M ,dupper, dlower,sense
    work->scaling= malloc(m*sizeof(c_float));
    for(i =0; i < ms; i++) work->scaling[i] =1;
    work->M = malloc(n*(m-ms)*sizeof(c_float));
    work->dupper = malloc(m*sizeof(c_float));
    work->dlower = malloc(m*sizeof(c_float));
    work->sense = malloc(m*sizeof(int));

    // Allocate memory for Rinv
    work->Rinv = (alloc_R == 1) ? malloc(((n+1)*n/2)*sizeof(c_float)) : NULL;
    // Allocate memory for v
    work->v = (alloc_v == 1) ? malloc(n*sizeof(c_float)) :  NULL;

#ifdef SOFT_WEIGHTS
    // Allocate memory for soft weights
    work->d_ls = malloc(m*sizeof(c_float));
    work->d_us = malloc(m*sizeof(c_float));
    work->rho_ls= malloc(m*sizeof(c_float));
    work->rho_us= malloc(m*sizeof(c_float));
    for(i = 0; i< m; i++){
        work->d_ls[i] = 0;
        work->d_us[i] = 0;
        work->rho_ls[i] = DEFAULT_RHO_SOFT;
        work->rho_us[i] = DEFAULT_RHO_SOFT;
    }
#endif
}


// Free memory for iterates
void free_daqp_workspace(DAQPWorkspace *work){
    if(work->lam != NULL){
        free(work->lam);
        free(work->lam_star);

        free(work->WS);

        free(work->L);
        free(work->D);

        free(work->xldl);
        free(work->zldl);

        free(work->u);

        free(work->xold);

        work->lam = NULL;
    }

    if(work->settings != NULL){ 
        free(work->settings);
        work->settings = NULL;
    }

    free_daqp_bnb(work);

}

// Extract solution information from workspace 
void daqp_extract_result(DAQPResult* res, DAQPWorkspace* work){
    int i; 
    // Extract primal solution
    for(i=0;i<work->n;i++) res->x[i] = work->x[i];

    // Extract dual solution
    if(res->lam != NULL){
        for(i=0;i<work->m;i++) 
            res->lam[i] = 0; 
        for(i=0;i<work->n_active;i++)
            res->lam[work->WS[i]] = work->lam_star[i];
    }

    // Shift back function value
    if(work->v != NULL && (work->settings->eps_prox == 0
                || work->Rinv != NULL || work->RinvD != NULL)){ // Normal QP
        res->fval = work->fval;
        for(i=0;i<work->n;i++) res->fval-=work->v[i]*work->v[i];
        res->fval *=0.5;
        if(work->settings->eps_prox != 0)
            for(i=0;i<work->n;i++) // compensate for proximal iterations
                res->fval+= work->settings->eps_prox*work->x[i]*work->x[i];
    }
    else if(work->qp != NULL && work->qp->f != NULL ){ // LP
        res->fval = 0;
        for(i=0;i<work->n;i++) res->fval+=work->qp->f[i]*work->x[i];
    }

    // info
    res->soft_slack = work->soft_slack;
    res->iter = work->iterations;
    res->nodes = (work->bnb == NULL) ? 1 : work->bnb->nodecount;
}

void daqp_default_settings(DAQPSettings* settings){
    settings->primal_tol = DEFAULT_PRIM_TOL;
    settings->dual_tol = DEFAULT_DUAL_TOL; 
    settings->zero_tol = DEFAULT_ZERO_TOL;
    settings->pivot_tol = DEFAULT_PIVOT_TOL;
    settings->progress_tol= DEFAULT_PROG_TOL;

    settings->cycle_tol = DEFAULT_CYCLE_TOL;
    settings->iter_limit = DEFAULT_ITER_LIMIT;
    settings->fval_bound = DAQP_INF; 

    settings->eps_prox = 0;
    settings->eta_prox = DEFAULT_ETA;

    settings->rho_soft = DEFAULT_RHO_SOFT; 

    settings->rel_subopt = DEFAULT_REL_SUBOPT;
    settings->abs_subopt = DEFAULT_ABS_SUBOPT;

    settings->sing_tol = DEFAULT_SING_TOL;
    settings->refactor_tol = DEFAULT_REFACTOR_TOL;
}

/* Remove redundant constraints*/
// Determine reundant constraint of the polyhedron A*x <= b
void daqp_minrep(int* is_redundant, c_float* A, c_float* b, int n, int m, int ms)
{
    int i;
    // Setup workspace
    DAQPWorkspace work;
    work.settings = NULL;
    allocate_daqp_workspace(&work,n,0);
    allocate_daqp_settings(&work);
    work.M = A;
    work.dupper = b;
    work.m = m;
    work.ms = ms;

    work.dlower = malloc(m*sizeof(c_float));
    work.sense = malloc(m*sizeof(int));
    for(i = 0; i<m; i++){
        work.dlower[i] = -DAQP_INF;
        work.sense[i] = 0;
    }


    // Solve minrep
    daqp_minrep_work(is_redundant, &work);
    // free
    free_daqp_workspace(&work);
    free(work.dlower);
    free(work.sense);
}
