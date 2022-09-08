//
// Created by carlos on 06/03/19.
//

#include <chrono>
#include <cmath>
#include "../headers/Model.h"

Model::Model(Graph *graph, int relaxNum, bool heuristics, bool barrierMethod, double lambda, int maxIter, int B, int time)
{
  Model::graph = graph;
  Model::lambda = lambda;
  Model::maxIter = maxIter;
  Model::B = B;
  Model::time = time;
  Model::relaxNum = relaxNum;
  Model::heuristics = heuristics;
  Model::barrierM = barrierMethod;

  LB = 0, UB = 0, iter = 0;
  if (heuristics)
    heuristic = new Heuristic(graph);
}

void Model::initialize()
{
  int o, d, n = graph->getN(), m = graph->getM();
  try
  {

    env.set("LogFile", "MS_mip.log");
    env.start();

    y = vector<vector<GRBVar>>(n, vector<GRBVar>(n));
    xi = vector<GRBVar>(n);
    z = vector<GRBVar>(n);
    lambda_delay = vector<GRBVar>(n);

    char name[30];
    for (o = 0; o < n; o++)
    {
      for (auto *arc : graph->arcs[o])
      {
        d = arc->getD();
        sprintf(name, "y_%d_%d", o, d);
        y[o][d] = model.addVar(0.0, 1.0, 0, GRB_BINARY, name);
      }
    }

    for (auto i : graph->terminals)
    {
      sprintf(name, "z_%d", i);
      z[i] = model.addVar(0.0, 1.0, 0, GRB_BINARY, name);
    }

    for (int i = 0; i < n; i++)
    {
      sprintf(name, "lambda_%d", i);
      lambda_delay[i] = model.addVar(0.0, graph->getParamDelay() + 1, 0.0, GRB_CONTINUOUS, name);
      sprintf(name, "xi_%d", i);
      xi[i] = model.addVar(0.0, graph->getParamJitter() + 1, 0.0, GRB_CONTINUOUS, name);
    }

    delta_min = model.addVar(1.0, graph->getParamDelay() - graph->getParamVariation(), 0.0, GRB_CONTINUOUS, "delta_min");

    model.update();
    cout << "Create variables" << endl;
  }
  catch (GRBException &ex)
  {
    cout << ex.getMessage() << endl;
    cout << ex.getErrorCode() << endl;
    exit(EXIT_FAILURE);
  }
}

void Model::initModelArb()
{
  cout << "Begin the model creation" << endl;
  objectiveFunction();
  allNodesAttended();
  calcLambdaXi();
  cout << "All done!" << endl;
}

void Model::allNodesAttended()
{
  model.addConstr(y[graph->getRoot()][0] == 1);

  for (auto j : graph->DuS)
  {
    if (graph->removed[j])
    {
      model.addConstr(y[0][j] == 1, "in_arcs_" + to_string(j));
      model.addConstr(lambda_delay[j] == graph->getParamDelay() + 1, "lambda_" + to_string(0) + "_" + to_string(j));
      model.addConstr(xi[j] == graph->getParamJitter() + 1, "xi_" + to_string(0) + "_" + to_string(j));
    }
    else
    {
      GRBLinExpr inArcs;
      for (int i = 0; i < graph->getN(); i++)
      {
        for (auto *arc : graph->arcs[i])
        {
          if (arc->getD() == j)
          {
            inArcs += y[i][j];
          }
        }
      }
      model.addConstr(inArcs == 1, "in_arcs_" + to_string(j));
    }
  }
  model.update();
  cout << "All nodes are inserted" << endl;
}

void Model::calcLambdaXi()
{
  model.addConstr(delta_min <= graph->getParamDelay() - graph->getParamVariation());
  model.addConstr(delta_min >= 1);
  int j, bigM;
  for (int i = 0; i < graph->getN(); i++)
  {
    for (auto *arc : graph->arcs[i])
    {
      j = arc->getD();
      bigM = graph->getParamDelay() + 1;
      model.addConstr(lambda_delay[j] >= (lambda_delay[i] + (arc->getDelay() * y[i][j])) - (bigM * (1 - y[i][j])), "lambda_" + to_string(i) + "_" + to_string(j));
      model.addConstr(lambda_delay[j] <= (lambda_delay[i] + arc->getDelay()) + (bigM * (1 - y[i][j])), "lambda_minus_" + to_string(i) + "_" + to_string(j));

      bigM = graph->getParamJitter() + 1;
      model.addConstr(xi[j] >= xi[i] + (arc->getJitter() * y[i][j]) - (bigM * (1 - y[i][j])), "xi_" + to_string(i) + "_" + to_string(j));
    }
  }
  model.update();
  cout << "Computing lambda and xi values" << endl;
}

void Model::objectiveFunction()
{
  int paramDelay = graph->getParamDelay(), paramJitter = graph->getParamJitter(), paramVariation = graph->getParamVariation();

  GRBLinExpr objective = 0;
  // model.setObjective(objective, GRB_MINIMIZE);

  for (auto k : graph->terminals)
  {
    objective += (z[k] +
                  multDelay[k] * (lambda_delay[k] - (paramDelay + z[k])) +
                  multJitter[k] * (xi[k] - (paramJitter + z[k])) +
                  multVarLeq[k] * ((delta_min - paramDelay * z[k]) - lambda_delay[k]) +
                  multVarGeq[k] * (lambda_delay[k] - (delta_min + paramVariation + (paramDelay * z[k]))));
  }

  model.setObjective(objective, GRB_MINIMIZE);

  model.update();
  cout << "Objective Function was added successfully!" << endl;
}

bool Model::solve()
{
  try
  {
    model.set("TimeLimit", "3600.0");
    model.set("OutputFlag", "0");
    model.update();
    model.write("model.lp");
    model.optimize();
    return true;
  }
  catch (GRBException &ex)
  {
    cout << ex.getMessage() << endl;
    return false;
  }
}

void Model::getGradientDelay(vector<double> &gradientDelay)
{
  int i, j, paramDelay;
  for (auto k : graph->terminals)
  {
    gradientDelay[k] = lambda_delay[k].get(GRB_DoubleAttr_X) - graph->getParamDelay();
    if (z[k].get(GRB_DoubleAttr_X) > 0.1)
      gradientDelay[k] -= graph->getBigMDelay();

    if (gradientDelay[k] > 0)
      feasible = false;
  }
}

void Model::getGradientJitter(vector<double> &gradientJitter)
{
  int i, j, paramJitter;
  for (auto k : graph->terminals)
  {
    gradientJitter[k] = xi[k].get(GRB_DoubleAttr_X) - graph->getParamJitter();
    if (z[k].get(GRB_DoubleAttr_X) > 0.1)
      gradientJitter[k] -= graph->getBigMJitter();
    if (gradientJitter[k] > 0)
      feasible = false;
  }
}

void Model::getGradientVar(vector<double> &gradientVarLeq, vector<double> &gradientVarGeq)
{
  int i, j, n = graph->getN();
  int paramDelay = graph->getParamDelay(), paramJitter = graph->getParamJitter(), paramVariation = graph->getParamVariation();

  for (int k : graph->terminals)
  {
    gradientVarLeq[k] = delta_min.get(GRB_DoubleAttr_X) - lambda_delay[k].get(GRB_DoubleAttr_X);
    gradientVarGeq[k] = lambda_delay[k].get(GRB_DoubleAttr_X) - delta_min.get(GRB_DoubleAttr_X) - paramVariation;

    if (z[k].get(GRB_DoubleAttr_X) > 0.1)
    {
      gradientVarLeq[k] -= paramDelay;
      gradientVarGeq[k] -= paramDelay;
    }

    if (gradientVarLeq[k] > 0 || gradientVarGeq[k] > 0)
      feasible = false;
  }
}

double Model::getNormDelaynJitter(vector<double> &gradient)
{
  double sum = 0;
  for (auto k : graph->terminals)
    sum += pow(gradient[k], 2);
  return sqrt(sum);
}

double Model::getNormVar(vector<vector<double>> &gradient)
{
  double sum = 0;
  for (int k : graph->terminals)
    for (int l : graph->terminals)
      if (k != l)
        sum += pow(gradient[k][l], 2);
  return sqrt(sum);
}

int Model::getOriginalObjValue()
{
  int foValue = 0;
  for (auto k : graph->terminals)
    if (z[k].get(GRB_DoubleAttr_X) > 0)
      foValue++;
  return foValue;
}

bool Model::isFeasible()
{
  if (feasible)
    return true;
  feasible = true;
  return false;
}

int Model::lagrangean()
{
  int progress = 0, iter = 0, n = graph->getN();
  double thetaDelay, normDelay, thetaJitter, normJitter, thetaVarLeq, thetaVarGeq, normVarLeq, normVarGeq, objPpl, originalObj;

  vector<double> gradientDelay = vector<double>(n);
  vector<double> gradientJitter = vector<double>(n);
  vector<double> gradientVarLeq = vector<double>(n);
  vector<double> gradientVarGeq = vector<double>(n);

  multDelay = vector<double>(n);
  multJitter = vector<double>(n);
  multVarLeq = vector<double>(n);
  multVarGeq = vector<double>(n);

  if (barrierM)
  {
    // auto start = chrono::steady_clock::now();

    // BarrierMethod *bm = new BarrierMethod(graph);
    // bm->initModel();
    // bm->solve();
    // if (relaxNum == 1 || relaxNum == 3)
    //   bm->getMultipliersDelay(multipliersDelay);
    // if (relaxNum <= 2)
    //   bm->getMultipliersJitter(multipliersJitter);

    // bm->getMultipliersVariation(multVar);

    // auto end = chrono::steady_clock::now();
    // bmTime = chrono::duration_cast<chrono::seconds>(end - start).count();
  }

  auto start = chrono::steady_clock::now();
  auto end = chrono::steady_clock::now();
  endTime = 0;

  if (heuristics)
    UB = heuristic->initialHeuristic();
  UB = 12;
  LB = numeric_limits<short>::min();

  firstUB = UB;
  iterBub = 0;

  initialize();
  initModelArb();

  while (iter < maxIter && endTime < time)
  {
    if (solve())
    {

      getGradientDelay(gradientDelay);
      getGradientJitter(gradientJitter);
      getGradientVar(gradientVarLeq, gradientVarGeq);

      objPpl = model.get(GRB_DoubleAttr_ObjVal);
      if (iter == 0)
        firstLB = objPpl;

      if (objPpl > LB)
        LB = objPpl, progress = 0, iterBlb = iter;
      else
      {
        progress++;
        if (progress == B)
        {
          lambda *= 0.9;
          progress = 0;
        }
      }

      originalObj = getOriginalObjValue();
      cout << "Original Obj: " << originalObj << ", Computed Obj: " << objPpl << endl;

      if (isFeasible() && originalObj < UB)
      {
        UB = originalObj, iterBub = iter;
        if ((UB - LB) < 1)
          cout << "Result: " << LB << " - " << UB << endl;
        return UB;
      }

      // Heuristic
      if (heuristics)
      {
        // int heuObj = heuristic->subgradientHeuristic(multipliersRel, multipliersLeaf);
        // cout << "Heuristic Obj: " << heuObj << endl;
        // if (heuObj < UB)
        // {
        //   UB = heuObj, iterBub = iter;
        //   if ((UB - LB) / UB <= 0.0001)
        //     return UB;
        // }
      }

      normJitter = getNormDelaynJitter(gradientJitter);
      if (normJitter == 0)
        thetaJitter = 0;
      else
        thetaJitter = lambda * ((UB - objPpl) / pow(normJitter, 2));

      normDelay = getNormDelaynJitter(gradientDelay);
      if (normDelay == 0)
        thetaDelay = 0;
      else
        thetaDelay = lambda * ((UB - objPpl) / pow(normDelay, 2));

      normVarLeq = getNormDelaynJitter(gradientVarLeq);
      if (normVarLeq == 0)
        thetaVarLeq = 0;
      else
        thetaVarLeq = lambda * ((UB - objPpl) / pow(normVarLeq, 2));

      normVarGeq = getNormDelaynJitter(gradientVarGeq);
      if (normVarGeq == 0)
        thetaVarGeq = 0;
      else
        thetaVarGeq = lambda * ((UB - objPpl) / pow(normVarGeq, 2));

      for (int k : graph->terminals)
      {
        multDelay[k] = max(0.0, multDelay[k] + (gradientDelay[k] * thetaDelay));
        multJitter[k] = max(0.0, multJitter[k] + (gradientJitter[k] * thetaJitter));
        multVarLeq[k] = max(0.0, multVarLeq[k] + (gradientVarLeq[k] * thetaVarLeq));
        multVarGeq[k] = max(0.0, multVarGeq[k] + (gradientVarGeq[k] * thetaVarGeq));
      }

      cout << "(Feasible) Upper Bound = " << UB << ", (Relaxed) Lower Bound = " << LB << endl;
      objectiveFunction();

      iter++;
      end = chrono::steady_clock::now();
      endTime = chrono::duration_cast<chrono::seconds>(end - start).count();
      // getchar();
    }
  }

  return UB;
}

void Model::showSolution(string instance, int prepTime)
{
  ofstream output;
  output.open(instance, ofstream::app);

  output << "Prep. Time: " << prepTime << endl;
  output << "First LB: " << firstLB << "\nLB: " << LB << "\nIter. LB: " << iterBlb << endl;
  output << "First UB: " << firstUB << "\nUB: " << UB << "\nIter. UB: " << iterBub << endl;

  if (LB < 0)
    LB = 0;

  output << "gap: " << 100 * (double(UB - ceil(LB)) / double(UB)) << endl;
  output << "BM. Time: " << bmTime << "\nRuntime: " << endTime << endl;

  output.close();
}
