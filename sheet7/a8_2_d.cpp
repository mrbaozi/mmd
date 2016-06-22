TVector CMyFoldCanvas::Bayes(const int Nb,TVector Nmeas, TMatrix Pec, const float chi_cut, char* mode)
{

    printf("D'Agostini based unfolding called \n");
    printf("================================= \n");
    printf("Number of Bins = %d \n",Nb);

    // unfold distribution given in Nmeas(Nb) using the migration probabilities in Pec(Nb x Nb)
    // This is an implementation of the D'Agostini DESY 94-099 report about unfolding based on
    // Bayes theorem
    // THIS VERSION IS ONLY SUITABLE FOR 1D-HISTOGRAMS

    int i,j,l;
    int max_iter=200;

    // open Canvas for intermediate distributions
    TCanvas* cb = new TCanvas("BayesIter","Intermediate distribution",200,10,600,300);
    cb->SetGrid();
    cb->SetFillColor(42);
    // pad for plot
    TPad* padb = new TPad("BayesPad","Pad for intermediate distribution",0.02,0.02,0.98,0.98,21);
    padb->Draw();
    padb->cd();
    padb->SetGridx();
    padb->SetGridy();
    TH1F* h1fb = new TH1F("BayesIter","Intermediate Distribution",Nb,0.0,float(Nb));
    h1fb->SetMarkerStyle(21);
    h1fb->SetMarkerSize(0.7);
    h1fb->SetFillColor(5);
    h1fb->Draw();

    double Nobs=Nmeas.Norm1();  // number of observed events
    // set up initial probablities for the bins to uniform
    TVector  P0(Nb);
    P0 +=1./(double)Nb;
    TVector  N0(Nb);
    N0 = P0;
    N0 *= Nobs;

    // set up the efficiency vector
    TVector eps(Nb);
    for(i=0;i<Nb;i++)
        for(j=0;j<Nb;j++) eps(i) += Pec(j,i);
    double chi2,pnorm,Ntrue;
    TMatrix  Pce(Nb,Nb);
    // set up Vector for unfolded event numbers per bin
    TVector  NC(Nb);
    TVector  PC(Nb);
    int  iter=1;
    do{
        // show current distribution N0
        if(iter<5)
        {
            for(i=0;i<Nb;i++) h1fb->SetBinContent(i,N0(i));
            h1fb->Draw();
            cb->Modified();
            cb->Update();
            getchar();
        }
        // determine matrix Pce(nb,nb)
        for(i=0;i<Nb;i++)
        {
            for(j=0;j<Nb;j++)
            {
                Pce(i,j) = Pec(j,i)*P0(i);
                pnorm = 0.0;
                for(l=0;l<Nb;l++) pnorm += Pec(j,l)*P0(l);
                Pce(i,j) = Pce(i,j)/pnorm;
            }
        }
        // calculate the true events
        for(i=0;i<Nb;i++)
        {
            NC(i) = 0.0;
            for(j=0;j<Nb;j++)
                NC(i) += Nmeas(j)*Pce(i,j)/eps(i);
        }
        Ntrue = NC.Norm1(); // total number of expected events
        PC = NC;
        PC *= 1./Ntrue;      // get new normalized distribution
        //calculate the chi**2
        chi2 = 0.0;
        for(i=0;i<Nb;i++)
            if(N0(i)>0.0) chi2 += pow(NC(i)-N0(i),2)/N0(i);
        printf("Iteration %d :  chi2 = %f \n",iter++,chi2);
        N0 = NC;
        P0 = PC;
    } while(chi2 > chi_cut && iter<max_iter);
    delete padb;
    delete cb;
    delete h1fb;

    return NC;
}

