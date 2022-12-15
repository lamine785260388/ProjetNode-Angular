import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { Devise } from './class/devise';
import { findonePays_Devices } from './class/PaysDeviseFindOne';

@Injectable({
  providedIn: 'root'
})
export class AllServicesService {

  constructor(private router:Router,private http: HttpClient) { }
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };
InsertTraitement(data:any){
     return  this.http
     .post<any>(
       "http://localhost:3000/api/InsertTransaction",
       { data },
       this.httpOptions
     )
  }
  findOnePays_Devise(data:any|Devise|findonePays_Devices):any|Devise|findonePays_Devices{
return   this.http
.post<any|Devise|findonePays_Devices>(
  "http://localhost:3000/api/findonePays_Devices",{data},
  this.httpOptions
  )
  }
}
