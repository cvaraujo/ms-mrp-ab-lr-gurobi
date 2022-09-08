//
// Created by carlos on 06/03/19.
//

#ifndef MRP_MODEL_H
#define MRP_MODEL_H

#include "Include.h"
#include "Graph.h"
#include "Heuristic.h"
#include "BarrierMethod.h"

class Model
{
  Graph *graph;
  GRBEnv env = GRBEnv();
  GRBModel model = GRBModel(env);
  vector<vector<GRBVar>> y;
  vector<GRBVar> z, xi;
  vector<GRBVar> lambda_delay;
  GRBVar delta_min;
  vector<double> multDelay, multJitter, multVarLeq, multVarGeq;
  int beginConstrRel = 0, endConstrRel = 0, beginConstrVar = 0, endConstrVar = 0;
  bool heuristics = false, barrierM, feasible = true;
  int firstUB, iter, iterBlb = 0, iterBub = 0, relaxNum, B, time, preprocessingTime = 0, maxIter = 1000, UB, bmTime = 0, endTime = 0;
  double firstLB, LB, lambda = 1.5;
  Heuristic *heuristic;

  void initialize();

  void initModelArb();

  void preprocessing();

  void changeCoefObjective();

  void objectiveFunction();

  void allNodesAttended();

  void calcLambdaXi();

  void getGradient(vector<double> &gradientDelay, vector<double> &gradientJitter);

  void getGradientDelay(vector<double> &gradientDelay);

  void getGradientJitter(vector<double> &gradientJitter);

  void getGradientVar(vector<double> &gradientVarLeq, vector<double> &gradientVarGeq);

  double getNormDelaynJitter(vector<double> &gradient);

  double getNormVar(vector<vector<double>> &gradient);

  int getOriginalObjValue();

  bool isFeasible();

  void objectiveFunctionLrArb();

public:
  Model(Graph *graph, int relaxNum, bool heuristics, bool barrierMethod, double lambda, int maxIter, int B, int time);

  void initializeLinear();

  void initModel();

  void initModelLinearRelaxation();

  void initModelCshp();

  bool solve();

  void solveLinear();

  void showSolution(string instance, int prepTime);

  int lagrangean();
};

#endif // MRP_MODEL_H
