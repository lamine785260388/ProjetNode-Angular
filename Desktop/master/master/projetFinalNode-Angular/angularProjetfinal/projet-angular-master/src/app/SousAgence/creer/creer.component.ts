import { SousAgence } from './../../class/sous-agence';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { AllServicesService } from 'src/app/all-services.service';
import { Agence } from 'src/app/class/agence';
import Swal from 'sweetalert2';


@Component({
  selector: 'app-creer',
  templateUrl: './creer.component.html',
  styleUrls: ['./creer.component.css']
})
export class CreerComponent implements OnInit {
  donneAgence:Agence[] ;

  constructor(private http:HttpClient,private serviceSousAgence:AllServicesService){

  }
  ngOnInit(): void {
    this.serviceSousAgence.getAgence()
    .subscribe(res=>{
this.donneAgence=res.data
console.log(this.donneAgence)
    })
  }
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };



  submit (form: NgForm) {
    //username password idSousAgence
    let code_sous_agence=form.value.code_sous_agence
    let nom_sous_agence=form.value.nom_sous_agence

    let  addresse_sous_agence=form.value. addresse_sous_agence
    let city_sous_agence=form.value.city_sous_agence
    let country_sous_agence=form.value.country_sous_agence
    let phone_sous_agence=form.value.phone_sous_agence
    let email_sous_agence=form.value.email_sous_agence
    let AGENCEId=form.value.AGENCEId

    this.serviceSousAgence.InsertSousAgence(code_sous_agence,nom_sous_agence,addresse_sous_agence,city_sous_agence,country_sous_agence, phone_sous_agence,email_sous_agence,AGENCEId)

    .subscribe((res) =>{
      if(res.erreur==false){
        Swal.fire(
          res.message,
          'Avec succés',
          'success'
        )
      }


  }


    )

  }

}
