import { Pays } from './class/pays';

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Router } from '@angular/router';
@Injectable()
export class Service {
    constructor(private http: HttpClient,private router:Router) {

    }
    httpOptions = {
        headers: new HttpHeaders({
          "Content-Type": "application/json"
        })
      };
    getAllPays(){
         this.http
        .get<Pays>(
          "http://localhost:3000/api/findAllPays",
          );
      }
  
}
