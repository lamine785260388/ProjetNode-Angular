
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Router } from '@angular/router';

@Injectable()
export class MesService {
    constructor(private router:Router,private http: HttpClient){}
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
  
}
