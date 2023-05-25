import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { Devise } from './class/devise';
import { findonePays_Devices } from './class/PaysDeviseFindOne';
import { Observable } from 'rxjs';
import { Transaction } from './class/transaction';
import { BaseUrl } from './class/baseurl';

@Injectable({
  providedIn: 'root'
})
export class AllServicesService {

  constructor(private router:Router,private http: HttpClient) { }
  url=new BaseUrl;
  racine=this.url.url;
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };



  InsertAgence(code_agence:string,nom_agence:string,statut_agence:string):Observable<any>{
    return  this.http
    .post<any>(
      this.racine+"InsertAgence",
      { code_agence:code_agence,nom_agence:nom_agence,statut_agence:statut_agence },
      this.httpOptions
    )
  }
}
