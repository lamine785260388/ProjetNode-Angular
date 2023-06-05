import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { Devise } from './class/devise';
import { findonePays_Devices } from './class/PaysDeviseFindOne';
import { Observable } from 'rxjs';
import { Transaction } from './class/transaction';
import { BaseUrl } from './class/baseurl';
import { Agence } from './class/agence';

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
  InsertSousAgence(code_sous_agence:string, nom_sous_agence:string,addresse_sous_agence:string,city_sous_agence:string,country_sous_agence:string, phone_sous_agence:string,email_sous_agence:string,AGENCEId:Int16Array):Observable<any>{

    return  this.http
    .post<any>(
      this.racine+"InsertSousAgence",
      { code_sous_agence:code_sous_agence,nom_sous_agence:nom_sous_agence,addresse_sous_agence:addresse_sous_agence,city_sous_agence:city_sous_agence,country_sous_agence:country_sous_agence,phone_sous_agence:phone_sous_agence,email_sous_agence:email_sous_agence,AGENCEId:AGENCEId },
      this.httpOptions
    )
  }
  getAgence():Observable<Agence[]|any>{
    return this.http.get<Agence[]>(this.racine+"findAllAgence")
  }
}
