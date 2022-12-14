import { Router } from '@angular/router';
import { OnInit } from '@angular/core';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Service } from 'src/app/node.service';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Pays } from 'src/app/class/pays';

@Component({
  selector: 'app-send',
  templateUrl: './send.component.html',
  styleUrls: ['./send.component.css']
})
export class SendComponent implements OnInit {
constructor(private router:Router,private http: HttpClient){
 if(sessionStorage.getItem('isloggin')!='true'){
  sessionStorage.setItem('url','send')
  this.router.navigate(['login'])
 }
 }
 allPays:Pays[]|any
 
 httpOptions = {
  headers: new HttpHeaders({
    "Content-Type": "application/json"
  })
};
  ngOnInit(): void {

    console.log(sessionStorage.getItem('isloggin'))
    this.http
    .get<Pays[]|any>(
      "http://localhost:3000/api/findAllPays",
      )
      .subscribe(res=>{
this.allPays=res.data
console.log(this.allPays[0])
      })
   
   
  }
  
  submit (form: NgForm) {
     //recuperation des informations de l'emmeteur
    var prenom = form.value.prenomemetteur;
    var nom=form.value.nomemetteur;
    var cni=form.value.cniemetteur
    var phone=form.value.phoneemetteur

    this.http
      .post(
        "http://localhost:3000/api/InsertClient",
        { id: cni, nom_client: nom,prenom_client:prenom,phone:phone },
        this.httpOptions
      )
      .subscribe(res=>{
        console.log(res+"insertion passé avec succés")
      })
      var prenom = form.value.prenomrecepteur;
    var nom=form.value.nomrecepteur;
    var cni=form.value.cnirecepteur;
    var phone=form.value.phonerecepteur;
    this.http
      .post(
        "http://localhost:3000/api/InsertClient",
        { id: cni, nom_client: nom,prenom_client:prenom,phone:phone },
        this.httpOptions
      )
      .subscribe(res=>{
        console.log(res+"insertion recepteur")
      })
     
   }
}
