
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
#include "FloatMat.h"
#include "GFN_Model.h"


//
char LogFileNameGFN[256];
//
void createLogTrainingGFN()
{
	FILE * fp = fopen(LogFileNameGFN, "w");
	fclose(fp);
}
void printStrToLogTrainingGFN(char * str)
{
	FILE * fp = fopen(LogFileNameGFN,"a");
	fprintf(fp,"%s",str);
	fclose(fp);
}
//

// utility functions
int GFN_Predict(GFN_Model & gfn, FloatMat & Samples, FloatMat & Results)
{
	int NumLayers = gfn.getNumLayers();
	//
	int * ArrActs = new int[NumLayers];
	gfn.getArrayActs(ArrActs);
	//
	int NumMat = NumLayers - 1;
	//
	Results = Samples;
	//
	for (int i = 0; i < NumMat; i++)
	{
		Results = Internal_Activiation_GFN(Results * gfn.Weights[i], gfn.Shifts[i], ArrActs[i]);
	}
	//
	//Result.normalizeRows();

	//
	delete [] ArrActs;

	return 1;
}
//
int GFN_Train(GFN_Model & gfn, FloatMat & Samples, FloatMat & Labels)
{
	// return positive for trained as scheduled,
	// return negative for training error,
	// return 0 for not trained,

	if (gfn.FlagLearningMethod == gfn.LEARN_GD)
	{
		return FunctionGFN_Train_GD(gfn, Samples, Labels);
	}
	else if (gfn.FlagLearningMethod == gfn.LEARN_LM)
	{
		return FunctionGFN_Train_LM(gfn, Samples, Labels);
	}
	else
	{
		printf("gfn.FlagTrainingMethod %d NOT defined.\n", gfn.FlagLearningMethod);
		//
		return 0;
	}
}
//
int GFN_Test(GFN_Model & gfn, FloatMat & Samples, FloatMat & Labels)
{
	gfn.ValidationSamples.copyFrom(Samples);
	gfn.ValidationLabels.copyFrom(Labels);

	return Internal_ValidationCheck_GFN(gfn);
}
//

// training functions
int FunctionGFN_Train_GD(GFN_Model & gfn, FloatMat & Samples, FloatMat & Labels)
{
	// return positive for trained as scheduled,
	// return negative for training error,

	// gdd, gda, gdm, gdx,
	float alpha = gfn.alpha;
	float beta = gfn.beta;
	float delta = gfn.delta;
	//
	float lamda = gfn.lamda;   // momentum
	float lamda_m = 1 - lamda;
	//
	int MaxIter = gfn.MaxIter;
	float error_tol = gfn.error_tol;
	float gradient_tol = gfn.gradient_tol;
	//
	float alpha_threshold = gfn.alpha_threshold;
	//
	//
	int FlagErrBalance = gfn.FlagErrBalance;
	//int FlagLearningMethod = gfn.FlagLearningMethod;
	int FlagAlpha = gfn.FlagAlpha;
	int FlagMomentum = gfn.FlagMomentum;
	//
	//float learning_portion = gfn.learning_portion;  // not used here
	//int SeedLearning = gfn.SeedLearning;
	//

	// log
	char StrTemp[256];
	//
	sprintf(LogFileNameGFN, "LogTrainingGD_%d_%d_%d_%.4f_%.4f_%.6f_%.2f_%d_%d.txt",
			FlagErrBalance, FlagAlpha, FlagMomentum, alpha, beta, delta, lamda, MaxIter, gfn.SeedLearning);
	//
	createLogTrainingGFN();
	//

	//
	printf("TrainingStart.\n");
	//

	// 训练样本划分
	printf("Dividing Samples ...\n");
	//
	int iResult = Internal_DivideSamples_GFN(gfn, Samples, Labels);
	if (iResult < 0)
	{
		printf("Error: too few samples, or not fit for the structure of the model.\n");
		sprintf(StrTemp, "Error: too few samples, or not fit for the structure of the model.\n");
		printStrToLogTrainingGFN(StrTemp);

		return -1;
	}
	//
	while (iResult == 0)   //
	{
		printf("Divided not properly.\n");
		printf("Dividing Samples ...\n");
		//
		iResult = Internal_DivideSamples_GFN(gfn, Samples, Labels);
	}
	printf("Samples Divided.\n");
	//

	// 步长
	float alpha_t = alpha;
	//
	// 误差
	float err;
	float gradient_length;
	//
	float err_last = 100000;
	//

	// 误差平衡
	FloatMat MatErrBalance;
	//
	int NumSamples, NumTypes;
	gfn.LearningLabels.getMatSize(NumSamples, NumTypes);
	//
	if (FlagErrBalance == 1)
	{
		if (Internal_MatErrBalance_GFN(gfn, MatErrBalance) == 0)   //
		{
			printf("Error: MatErrBalance failed to be generated.\n");
			sprintf(StrTemp, "Error: MatErrBalance failed to be generated.\n");
			printStrToLogTrainingGFN(StrTemp);

			return -3;
		}
	}
	else if (FlagErrBalance != 0)
	{
		printf("Error: FlagErrBalance != 0 or 1.\n");
		sprintf(StrTemp, "Error: FlagErrBalance != 0 or 1.\n");
		printStrToLogTrainingGFN(StrTemp);

		return -2;

	}// if FlagErrBalance


	// 网络结构
	int NumLayers = gfn.getNumLayers();
	int * ArrayNumNodes = new int[NumLayers];
	gfn.getArrayNumNodes(ArrayNumNodes);
	//
	int * ArrayActs = new int[NumLayers];
	gfn.getArrayActs(ArrayActs);
	//
	int NumMat = NumLayers - 1;
	int NumM2 = NumMat - 1;
	//

	// 中间结果，求导时要用，
	FloatMat * ActDerivative = new FloatMat[NumMat];
	FloatMat * Results = new FloatMat[NumLayers];
	//
	Results[0].copyFrom(gfn.LearningSamples);

	// 误差与回溯的矩阵，
	FloatMat ErrTrack, TrackAct;

	// 导数
	FloatMat * GradientW = new FloatMat[NumMat];
	FloatMat * GradientS = new FloatMat[NumMat];
	//
	for (int layer = 0; layer < NumMat; layer++)
	{
		GradientW[layer].setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
		GradientS[layer].setMatSize(1, ArrayNumNodes[layer+1]);
	}

	// 惯性
	FloatMat * deltaW;
	FloatMat * deltaS;
	float * ptr_momentum_swap;
	//
	if (FlagMomentum == gfn.MOMENTUM_NONE)
	{
		deltaW = new FloatMat[1];
		deltaS = new FloatMat[1];
	}
	else
	{
		deltaW = new FloatMat[NumMat];
		deltaS = new FloatMat[NumMat];
		//
		for (int layer = 0; layer < NumMat; layer++)
		{
			deltaW[layer].setMatSize(ArrayNumNodes[layer], ArrayNumNodes[layer+1]);
			deltaS[layer].setMatSize(1, ArrayNumNodes[layer+1]);

			deltaW[layer].setMatConstant(0);
			deltaS[layer].setMatConstant(0);
		}
	}

	//
	int iRet = 1;     // MaxIter reached
	//

	// 循环优化
	int iter = 0;
	while (iter < MaxIter)
	{
		//
		printf("iter, %d, ", iter);
		sprintf(StrTemp, "iter, %d, ", iter);
		printStrToLogTrainingGFN(StrTemp);
		//

		// 前向计算
		for (int layer = 0; layer < NumMat; layer++)
		{
			Results[layer+1] = Results[layer] * gfn.Weights[layer];
			//
			ActDerivative[layer] = Internal_ActDerivative_GFN(Results[layer+1], gfn.Shifts[layer], ArrayActs[layer]);
			//
			Results[layer+1] = Internal_Activiation_GFN(Results[layer+1], gfn.Shifts[layer], ArrayActs[layer]);
		}

		// 误差计算
		ErrTrack = Results[NumMat] - gfn.LearningLabels;
		//
		if (FlagErrBalance == 1)
		{
			ErrTrack = ErrTrack.mul(MatErrBalance);
		}
		//
		err = ErrTrack.mul(ErrTrack).meanElementsAll();
		err = sqrt(err);
		//
		printf("err, %.4f, ", err);
		sprintf(StrTemp, "err, %.4f, ", err);
		printStrToLogTrainingGFN(StrTemp);
		//
		if (err < error_tol)
		{
			printf("err < error_tol = %f\n", error_tol);

			iRet = 2;  // error_tol reached
			break;
		}
		//

		// validation check
		Internal_ValidationCheck_GFN(gfn);
		//
		printf("prc, %.4f, ", gfn.performance[0]);
		sprintf(StrTemp, "prc, %.4f, ", gfn.performance[0]);
		printStrToLogTrainingGFN(StrTemp);
		//
		printf("rec, %.4f, ", gfn.performance[1]);
		sprintf(StrTemp, "rec, %.4f, ", gfn.performance[1]);
		printStrToLogTrainingGFN(StrTemp);
		//

		// 计算导数
		for (int layer = NumM2; layer >= 0; layer--)
		{
			// TrackAct
			TrackAct = ErrTrack.mul(ActDerivative[layer]);

			// 计算ErrTrack前一层
			ErrTrack = TrackAct * gfn.Weights[layer].transpose();

			// 计算Weights导数
			GradientW[layer] = Results[layer].transpose() * TrackAct; // + gfn.Weights[layer].getSigns() * epsilon;
			//
			GradientW[layer] = GradientW[layer].mul(gfn.Connects[layer]);

			// 计算Shifts导数
			GradientS[layer] = TrackAct.sumCols(); // + gfn.Shifts[layer].getSigns() * epsilon;

		}//for layer

		// 计算梯度长度
		gradient_length = 0;
		for (int layer = NumM2; layer >=0; layer--)
		{
			gradient_length += (GradientW[layer].mul(GradientW[layer])).sumElementsAll();  // 可优化
			gradient_length += (GradientS[layer].mul(GradientS[layer])).sumElementsAll();
		}
		gradient_length = sqrt(gradient_length);
		//
		printf("glength, %.4f, ", gradient_length);
		sprintf(StrTemp, "glength, %.4f, ", gradient_length);
		printStrToLogTrainingGFN(StrTemp);
		//
		if (gradient_length < gradient_tol)
		{
			printf("gradient_length < gradient_tol = %f\n", gradient_tol);

			iRet = 3; // gradient_tol reached
			break;
		}
		//

		// 梯度下降
		//
		if (alpha_t > alpha_threshold)
		{
			if (FlagAlpha == gfn.ALPHA_DES)    // 步长，下降
			{
				alpha_t *= beta;
			}
			else if (FlagAlpha == gfn.ALPHA_ADA)   // 步长，自适应
			{
				if (err < err_last) alpha_t += delta;
				else alpha_t *= beta;
				//
				err_last = err;
			}
		}
		//else alpha_t = alpha_threshold;
		//
		if (FlagMomentum == gfn.MOMENTUM_EXP)    // 动量，指数平滑，
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				deltaW[layer] = deltaW[layer] * lamda - GradientW[layer] * alpha_t;
				deltaS[layer] = deltaS[layer] * lamda - GradientS[layer] * alpha_t;
				//
				gfn.Weights[layer] = gfn.Weights[layer] + deltaW[layer];
				gfn.Shifts[layer] = gfn.Shifts[layer] + deltaS[layer];
			}
		}
		else if (FlagMomentum == gfn.MOMENTUM_PREV)    // 动量，两步合力，
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				gfn.Weights[layer] = gfn.Weights[layer] - (GradientW[layer] * (lamda * alpha_t) + deltaW[layer] * (lamda_m * alpha_t));
				gfn.Shifts[layer] = gfn.Shifts[layer] - (GradientS[layer] * (lamda * alpha_t) + deltaS[layer] * (lamda_m * alpha_t));

				//
				ptr_momentum_swap = deltaW[layer].data;
				deltaW[layer].data = GradientW[layer].data;
				GradientW[layer].data = ptr_momentum_swap;
				//
				ptr_momentum_swap = deltaS[layer].data;
				deltaS[layer].data = GradientS[layer].data;
				GradientS[layer].data = ptr_momentum_swap;
				//
			}
		}
		else // 无动量
		{
			for (int layer = NumM2; layer >= 0; layer--)
			{
				gfn.Weights[layer] = gfn.Weights[layer] - GradientW[layer] * alpha_t;
				gfn.Shifts[layer] = gfn.Shifts[layer] - GradientS[layer] * alpha_t;
			}
		}
		//

		//
		printf("alpha_t, %.6f,\n", alpha_t);
		sprintf(StrTemp, "alpha_t, %.6f\n", alpha_t);
		printStrToLogTrainingGFN(StrTemp);
		//

		//
		iter++;

	}// while iter
	//
	if (iter >= MaxIter)
	{
		printf("iter >= MaxIter = %d\n", MaxIter);
	}
	//

	//
	delete [] ArrayNumNodes;
	delete [] ArrayActs;
	delete [] ActDerivative;
	delete [] Results;
	//
	delete [] GradientW;
	delete [] GradientS;
	delete [] deltaW;
	delete [] deltaS;
	//

	return iRet;
}
//
int FunctionGFN_Train_LM(GFN_Model & gfn, FloatMat & Samples, FloatMat & Labels)
{
	printf("FunctionGFN_Train_LM() not implemented.\n");

	return 0;
}
//



// internal functions
FloatMat Internal_Activiation_GFN(FloatMat mat, FloatMat shift, int act)
{
	int NumRows, NumCols;
	mat.getMatSize(NumRows, NumCols);

	FloatMat answ(NumRows, NumCols);	

	float * answ_data_p = answ.data;
	float * mat_data_p = mat.data;
	float * shift_data_p = shift.data;

	//
	GFN_Model GFN;

	//
	if (act == GFN.ACT_LOGS)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				answ_data_p[Posi] = 1 /(1 + exp(-temp));
				//

				//
				Posi++;
			}
		}
	}
	else if (act == GFN.ACT_LOGB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				answ_data_p[Posi] = 2 /(1 + exp(-temp)) - 1;
				//

				//
				Posi++;
			}
		}
	}
	else if (act == GFN.ACT_RELB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < -1) answ_data_p[Posi] = -1;
				else if (temp > 1) answ_data_p[Posi] = 1;
				else answ_data_p[Posi] = temp;
				//

				//
				Posi++;
			}
		}

	}
	else // GFN_Model.ACT_RELU
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < 0) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = temp;
				//

				//
				Posi++;
			}
		}

	}// if act

	return answ;
}
//
FloatMat Internal_ActDerivative_GFN(FloatMat mat, FloatMat shift, int act)
{
	int NumRows, NumCols;
	mat.getMatSize(NumRows, NumCols);

	FloatMat answ(NumRows, NumCols);	

	float * answ_data_p = answ.data;
	float * mat_data_p = mat.data;
	float * shift_data_p = shift.data;

	//
	GFN_Model GFN;

	//
	if (act == GFN.ACT_LOGS)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function	
				// answ_data_p[Posi] = 1 /(1 + exp(-temp));
				//
				temp = exp(-temp);
				answ_data_p[Posi] = temp /(1 + temp)/(1 + temp);
				//

				//
				Posi++;
			}
		}
	}
	else if (act == GFN.ACT_LOGB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				// answ_data_p[Posi] = 2 /(1 + exp(-temp)) - 1;
				//
				temp = exp(-temp);
				answ_data_p[Posi] = 2 * temp /(1 + temp)/(1 + temp);
				//

				//
				Posi++;
			}
		}
	}
	else if (act == GFN.ACT_RELB)
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function				
				//
				if (temp < -1 || temp > 1) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = 1;
				//

				//
				Posi++;
			}
		}

	}
	else // GFN_Model.ACT_RELU
	{
		float temp;

		int Posi = 0;
		for (int i = 0; i < NumRows; i++)
		{
			for (int j = 0; j < NumCols; j++)
			{
				temp = mat_data_p[Posi] + shift_data_p[j];

				// activation function derivative			
				//
				if (temp < 0) answ_data_p[Posi] = 0;
				else answ_data_p[Posi] = 1;
				//

				//
				Posi++;
			}
		}

	}// if act

	return answ;
}
//

//
int Internal_DivideSamples_GFN(GFN_Model & gfn, FloatMat & Samples, FloatMat & Labels)
{
	// return negative for error
	// return 0 for divided not properly
	// return 1 for divided

	//
	int NumSamples, NumFeatures;
	int NumSamplesL, NumTypes;
	Samples.getMatSize(NumSamples, NumFeatures);
	Labels.getMatSize(NumSamplesL, NumTypes);
	//
	if (NumSamples < 10)   //
	{
		printf("Error: too few Samples.\n");

		return -1;
	}
	//
	if (NumSamples != NumSamplesL)
	{
		printf("Error: Samples and Labels do NOT have same number of rows.\n");

		return -2;
	}
	//
	int NumLayers = gfn.getNumLayers();
	int * ArrayNumNodes = new int[NumLayers];
	gfn.getArrayNumNodes(ArrayNumNodes);
	//
	if (NumFeatures != ArrayNumNodes[0])
	{
		printf("Error: Samples and Structure of the model do NOT match.\n");

		return -3;
	}
	//
	if (NumTypes != ArrayNumNodes[NumLayers-1])
	{
		printf("Error: Labels and Structure of the model do NOT match.\n");

		return -4;
	}

	// 随机选择
	int * FlagForLearning = new int[NumSamples];
	int NumLearning = 0;
	int NumValidation = 0;
	//
	srand(gfn.SeedLearning);
	int ThrRand = gfn.LearningPortion * 1000;  //
	//
	while (NumLearning == 0 || NumValidation == 0)
	{
		NumLearning = 0;
		NumValidation = 0;
		//
		for (int s = 0; s < NumSamples; s++)
		{
			if (rand()%1000 < ThrRand)  //
			{
				FlagForLearning[s] = 1;
				NumLearning++;
			}
			else
			{
				FlagForLearning[s] = 0;
				NumValidation++;
			}
		}// for s
	}// while 0

	// 复制
	gfn.LearningSamples.setMatSize(NumLearning, NumFeatures);
	gfn.LearningLabels.setMatSize(NumLearning, NumTypes);
	//
	gfn.ValidationSamples.setMatSize(NumValidation, NumFeatures);
	gfn.ValidationLabels.setMatSize(NumValidation, NumTypes);
	//
	float * data_learning_samples = gfn.LearningSamples.data;
	float * data_learning_labels = gfn.LearningLabels.data;
	float * data_validation_samples = gfn.ValidationSamples.data;
	float * data_validation_labels = gfn.ValidationLabels.data;
	//
	float * data_samples = Samples.data;
	float * data_labels = Labels.data;
	//
	int LenDatumSample = sizeof(float) * NumFeatures;
	int LenDatumLabel = sizeof(float) * NumTypes;
	//
	for (int s = 0; s < NumSamples; s++)
	{
		if (FlagForLearning[s] == 1)
		{
			memcpy(data_learning_samples, data_samples, LenDatumSample);
			data_learning_samples += NumFeatures;
			data_samples += NumFeatures;

			memcpy(data_learning_labels, data_labels, LenDatumLabel);
			data_learning_labels += NumTypes;
			data_labels += NumTypes;
		}
		else
		{
			memcpy(data_validation_samples, data_samples, LenDatumSample);
			data_validation_samples += NumFeatures;
			data_samples += NumFeatures;

			memcpy(data_validation_labels, data_labels, LenDatumLabel);
			data_validation_labels += NumTypes;
			data_labels += NumTypes;
		}
	}// for s

	//
	delete [] FlagForLearning;

	// 检查
	FloatMat MatCount;
	MatCount = gfn.ValidationLabels.sumCols();
	for (int t = 0; t < NumTypes; t++)
	{
		if (MatCount.data[t] == 0) return 0;
	}
	//
	MatCount = gfn.LearningLabels.sumCols();
	for (int t = 0; t < NumTypes; t++)
	{
		if (MatCount.data[t] == 0) return 0;
	}
	//

	return 1;
}
//
int Internal_MatErrBalance_GFN(GFN_Model & gfn, FloatMat & MatErrBalance)
{
	//0 for error, 1 for conducted,

	//
	FloatMat MatCount;
	MatCount = gfn.LearningLabels.sumCols();
	//

	//
	int NumSamples, NumTypes;
	gfn.LearningLabels.getMatSize(NumSamples, NumTypes);
	//
	float ratio = 1.0 * NumSamples/NumTypes;
	//
	float * data_count = MatCount.data;
	for (int i = 0; i < NumTypes; i++)
	{
		if (data_count[i] == 0)
		{
			printf("data_count[i] == 0 when generating MatErrBalance_GFN.\n");

			return 0;
		}
		//
		data_count[i] = ratio/data_count[i];
	}
	//

	//
	MatErrBalance.setMatSize(NumSamples, NumTypes);
	//
	int type = 0;
	float * data_label = gfn.LearningLabels.data;
	float * data_errbalance = MatErrBalance.data;
	//
	for (int s = 0, posi_start = 0; s < NumSamples; s++, posi_start += NumTypes)
	{
		// type
		for (int t = 0, posi = posi_start; t < NumTypes; t++, posi++)
		{
			if (data_label[posi] == 1)
			{
				type = t;
				break; // for t
			}
		}// for t

		// ratio
		ratio = data_count[type];

		// assign
		for (int t = 0, posi = posi_start; t < NumTypes; t++, posi++)
		{
			data_errbalance[posi] = ratio;
		}// for t

	}// for s

	//
	return 1;
}
//
int Internal_ValidationCheck_GFN(GFN_Model & gfn)
{
	FloatMat Results, ResultsNormalized;
	//
	GFN_Predict(gfn, gfn.ValidationSamples, Results);
	//
	ResultsNormalized.copyFrom(Results);
	ResultsNormalized.normalizeRows();
	//

	//
	int NumSamples, NumTypes;
	gfn.ValidationLabels.getMatSize(NumSamples, NumTypes);

	//
	float CriteriaAssertion = gfn.CriteriaAssertion;
	float CriteriaBasic = 1.0/NumTypes;
	//

	//
	int PositiveTotal = 0;
	int PredictedPositive = 0;
	int TruePredictedPositive = 0;
	//
	float * data_labels = gfn.ValidationLabels.data;
	float * data_results = Results.data;
	float * data_normalized = ResultsNormalized.data;
	//
	for (int s = 0; s < NumSamples; s++)
	{
		if (data_results[0] >= CriteriaBasic && data_normalized[0] >= CriteriaAssertion)
		{
			PredictedPositive++;

			if (data_labels[0] >= CriteriaAssertion)
			{
				TruePredictedPositive++;
			}
		}
		//
		if (data_labels[0] >= CriteriaAssertion)
		{
			PositiveTotal++;
		}

		//
		data_labels += NumTypes;
		data_results += NumTypes;
		data_normalized += NumTypes;

	}// for s

	//
	if (PredictedPositive > 0)	gfn.performance[0] = 1.0 * TruePredictedPositive/PredictedPositive;
	else gfn.performance[0] = 0;
	//
	gfn.performance[1] = 1.0 * TruePredictedPositive/PositiveTotal;
	//
	gfn.performance[2] = PositiveTotal;
	gfn.performance[3] = PredictedPositive;
	gfn.performance[4] = TruePredictedPositive;


	return 1;
}
//


